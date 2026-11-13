# warp 级调度扩展（2026-11-12）

## 问题

FA2 的手写版本里，warp w 自始至终拥有第 `16w .. 16w+15` 行：S 的 mma、行 max/sum、exp、PV 的 mma 与 O 都留在它的寄存器里。
现有的 tiling 调度里，每个 `group_tile_req` 独立地把结果分给 warp 或 work-item，请求之间只能经 shared memory 与 barrier 交换；
结果只能放在 shared 或每线程私有的 buffer，没有"按 mma 分片布局分在一个 warp 的寄存器里"这种位置；
warp 层只有 `warp_red`/`warp_mma`，没有分片上的逐元素、行归约与布局转换。
在每 SM 100 KB shared 的 GPU 上，S 与 O 占的 33 KB 让每 SM 最多 2 个块，参数怎么调都绕不过去（见 `bench/gpu_attn/RESULTS.zh-CN.md`）。

## 核心：warp 值是 32 元数组

一个 warp 的值的类型是 `«32; T»`，第 `l` 个分量是 lane `l` 持有的部分。
warp 级运算都写成 Mim 里的纯函数，它们就是规格：在 CPU 上可以直接求值，和普通数组程序一样做数值检查。
"每个 lane 只存自己的分量"是一种表示选择，由降级完成：

| 规格中的项 | warp 区域里 lane `l` 的表示 |
|---|---|
| `‹i: 32; e i›` | `e l` |
| `v#l` | `v` 的本地分量 |
| `v#(l xor c)`，`c` 是字面量 | `gpu.shfl_xor (v, c)` |
| `v#i`，`i` 是 lane 的其它函数 | `gpu.shfl_idx (v, i)` |

warp 程序的输入由 `gpu.lanes (self, x)` 给出：每个 lane 的本地值 `x` 组成的 warp 值，`self` 是本 lane。
读 pack 时 MimIR 的规范化会直接代入 pack 体，所以剩下的读只可能读 `gpu.lanes`、`tensor.mma_warp` 或 `tensor.warp_keep`。
`warp_keep` 是恒等，标记"共享算好的值"：读别的 lane 时 shuffle 该 lane 算出的结果，而不是代入定义重算；
没有加载 `gpu` 时它在规范化时消失，所以规格在 CPU 上照常求值。

降级只改变表示，不改变值，所以一个 warp 程序在 CPU 上的结果就是它在 GPU 上的结果；前提是 warp 内收敛（`gpu.outline` 已检查 warp 集体操作不在发散分支里）。

## 层次

1. **语义层（Mim，`tensor.warp`）**：一个 warp 拥有一个 16 行的块（多个块用数组）；warp 值、mma 分片（复用 `mma_m16n8k8_tf32` 的 `l_a/l_b/l_c` 与 `mma_ref`）、
   分片与 tile 互转的规格、分片上的逐元素、按行广播、行归约（规格与 quad 实现）、C 布局到 A 布局的置换。
2. **分布降级（`tensor.distribute`，在 `gpu.erase_phases` 之后）**：warp 区域内按上表把 `«32; T»` 的值换成本地分量与 shuffle。
3. **调度层**：warp 拥有的 fold：work-group → warp（拥有一个行块）→ 按 key 的 fold（状态是分片，在寄存器）→ 步内的 warp 级运算；
   K/V 面板仍由整个 work-group 协作填充，所以 warp 区域里有 barrier，所有 warp 必须执行同样多的步，由 `check_sync` 检查。
4. **TileFuse 生成**：把请求映射到语义层的运算（mma、逐元素、行归约），并让相邻请求沿用同一个行拥有者。

## 形式化

每条实现规则都写成"规格 = 实现"的引理，规格与实现都在 Mim 里：

- **分片互转**：`tile_of (frag_of t) = t`；`l_c` 是 `Idx 32 × Idx 4` 到 `Idx 16 × Idx 8` 的双射。
- **quad 行归约**：`l_c` 把第 `r` 行放在 lane `4 (r mod 8) .. 4 (r mod 8) + 3`，它们只在低两位不同；
  所以对可重组的律，先在本 lane 内归约、再 `xor 1`、`xor 2` 两次合并，等于按行归约的规格。
- **C 到 A 的置换**：收缩对 k 的任意置换不变；把 C 分片的 `(c0, c1, c2, c3)` 当作 A 的 `(c0, c2, c1, c3)`，
  等于把 key 按 `2t, 2t+1 ↦ t, t+4` 置换，B 按同一置换读即可。
- 能在 Mim 里对字面量求值的引理用 lit 测试检查（规格与实现在随机输入上逐元素比较）；
  对所有输入成立的论证写在本文，不能机械检查的部分标为可信边界。

## 里程碑

- **M1**（完成）：语义层与引理。`quad_covers_rows`、`c_to_a_permutes` 在编译期由 `refly.check` 穷举证明；
  `c_tile (c_of t) = t`、`row_fold` 等于规格、置换后的乘积等于原乘积在 CPU 上执行比较（`lit/tensor/warp_frag.mim`）。
- **M2**（完成）：分布降级；`lit/{gpu,nvptx}/warp_lanes.mim` 里一个 warp 用 `tensor.mma_warp` 与 `tensor.warp.row_fold` 算乘积与行最大值，
  降成一条 `mma_tf32` 与每个半行两次 `shfl_xor`，在 GPU 上对照参考结果。
- **M3a**（完成）：`bench/gpu_attn/flash_warp.py` 把手写 `flash_tc` 每一步的计算改写成 warp 程序
  （`tensor.mma_warp`、`row_fold`、`row_own` 与最后一次 `quad_merge`、`c_zip`/`c_rows`、`a_of_c` 与 `key_perm`），面板与循环仍手写。
  RTX PRO 6000 上与手写版本一致（BH × L = 32 × 2048：0.305 对 0.306 ms；128 × 2048：1.088 对 1.087 ms），结果正确：
  warp 代数的规格经分布降级就是手写的代码。
- **M3b**：调度层在 attention 上（warp 拥有的 fold、面板由 work-group 填充），对照 `flash_warp` 与 `attn_fa`。
- **M4**：TileFuse 自动生成。

## M3b：warp 拥有的区域

一个绑定到 warp 且标为"warp 拥有"的 `tiled` 区域（16 行一块），其中的张量不落到 buffer，而是 warp 值：
`16 × 8nt` 的张量是 `C nt`，长 16 的行向量是 `R`。LowerToMem 只做选择，把区域里的运算翻译成 `tensor.warp` 的 Mim 函数，
`tensor.distribute` 再把它们降成每个 lane 的代码：

| 区域里的运算 | 翻译 |
|---|---|
| 逐元素 `map_reduce`，输入是同布局的 warp 值、行向量或内存 | `c_zip`、`c_rows`，内存按 `c_pos` 逐 lane 读 |
| 对列的归约，可重组的律 | `row_fold`（或只在 lane 内的 `row_own`） |
| `16 × 8n × 8k` 的收缩 | `warp_mma` 链；A 是 C 布局的 warp 值时用 `a_of_c`，B 按 `key_perm` 读 |
| 写回区域的目标 | 每个 lane 按 `c_pos` 写自己的分量 |

W1（完成）：`tensor.bind 5` 标出 warp 拥有的区域；LowerToMem 的 `warp_term` 把区域里的张量翻译成 `tensor.warp` 的项
（`c_gen`/`r_gen` 给出每个位置的元素，`a_gen`/`b_gen`/`a_from_c`/`c_mma` 给出乘积），区域外的输入沿映射（`op_affine_map`）从内存读，
`store_warp` 让每个 lane 写自己的分量。`tensor.mma_warp`（原 `gpu.warp_mma`）没有 gpu 插件时就是 `mma_ref`。
`lit/{gpu,nvptx}/warp_owned.mim`：区域里的乘积与逐元素映射降成 32 条 `mma_tf32` 与寄存器上的运算，GPU 上结果正确。

W2（完成）：区域里的 `tile_fold` 由 `warp_fold` 发成循环，状态是每个 lane 的分量（`R` 为 `«2; F»`，`C nt` 为 `«nt, 4; F»`），
步内以 `gpu.lanes (self, x)` 作 warp 值；fold 的结果同样是 warp 值。
翻译也处理融进 `map_reduce_post` 的 epilogue 输入（按输出坐标读），以及融进收缩的 epilogue（逐位置作用在累加的 tile 上）；
收缩只接受 `acc + a · b`（`is_contraction`）。`lit/{gpu,nvptx}/warp_fold.mim`。
`bench/gpu_attn/attn_warp.mim`：attention 写成 work-group（`bind 1`）→ warp 拥有的 16 行（`bind 5`）→ fold，
不用 `group_tile_req`；K/V 由每个 lane 直接从全局内存读（W3 未做）。RTX PRO 6000 上 32 × 2048 为 0.590 ms
（`attn_fa` 0.827，手写 0.305），没有 shared 与 barrier，140 个寄存器。

W3（部分）：step 里的 `tensor.stage` 在 warp 拥有的区域由整个 work-group 协作填充进 shared（K/V 块的起点只依赖 work-group，
填充才是统一的），面板行尾填充 16 字节；共享缓冲不走 scratch，免得在 kernel 结尾被释放。
面板双缓冲：两组 shared 缓冲轮换，每步一道 barrier，下一块用 `cp.async` 在计算当前块时装载（复用 `pipeline_plan`）。
A 操作数来自内存且不随步变的乘积，其 A 分片在循环外装载一次、由循环携带（`hoisted_a_`）。

步骤：W1 上表的翻译；W2 区域里的 fold，状态在寄存器里跨步；W3 K/V 面板由 work-group 协作填充；W4 attention。
区域可以由用户显式写出，也可以由 TileFuse 的层级生成；W1 对两者相同，先用显式标记。

## 已知问题

- 带长度为 1 的轴的融合 Pack（如 `‹l: 32; ‹i: 1; ‹j: 4; …›››`）被下标读时，类型已折叠掉该轴而 Pack 的形状仍保留它，
  紧随的下标会剥错轴，产生 "expected `F`, got `«4; F»`" 的类型错误；语义层因此不带行块轴。核心问题未修。

## 16 位与 Hopper 原语（2026-11-13）

| 原语 | PTX | 与 warp 代数的关系 |
|---|---|---|
| `gpu.mma_f16`/`mma_bf16` | `mma.sync.m16n8k16`，F32 累加 | 规格 `tensor.mma_m16n8k16.mma_ref`；C 布局同 k8 |
| `gpu.ldmatrix_b16` | `ldmatrix(.trans).b16` | 非转置给出 A/K 的分片，转置给出行主序 B/V 的分片 |
| `gpu.wgmma_f16`/`bf16`，`wgmma_rs_*` | `wgmma.mma_async.m64nNk16` | warpgroup 的累加器 = 4 个 warp 各一个 `C nt`（行 `16 w ..`）；RS 的 A 是每个 warp 的 k16 A 分片 |
| `gpu.wgmma_fence/commit/wait` | `wgmma.fence/commit_group/wait_group` | `wait` 把累加器与寄存器操作数传过去，异步写在 SSA 上显式 |
| `gpu.wgmma_desc`（Mim 函数）、`gpu.shared_addr` | 矩阵描述符 | 128 B swizzle：行 `r` 的 16 字节块 `c` 存在 `c xor (r mod 8)` |
| `gpu.tensor_map`（host）、`gpu.tma_load` | `cuTensorMapEncodeTiled`、`cp.async.bulk.tensor.2d` | 产生与描述符相同的 swizzle 布局 |
| `gpu.mbar_*` | `mbarrier.*` | 生产者/消费者环的满/空信号 |
| `gpu.sync_named/arrive_named`、`gpu.setmaxnreg`、`gpu.fence_proxy_async` | `bar.sync/arrive id, n`、`setmaxnreg`、`fence.proxy.async` | warp 特化与乒乓 |

形式化：
- `tensor.mma_m16n8k16.c_pair_is_a`（`refly.check` 穷举）：两个相邻的 n8 C tile 就是一个 k16 A 分片，寄存器 `reg` 的第 `h` 半是 C tile `reg / 2` 的寄存器 `2 (reg mod 2) + h`。
  因此 16 位下 P 留在寄存器作 PV 的 A，不需要 TF32 时的 key 置换（`c_to_a_permutes`）；`a_of_c2` 是这一转换的规格。
- `layouts_cover`：k16 的 A、B 布局各覆盖其 tile 恰好一次。
- `mma_ref` 在 CPU 上与 `a_of_c2` 组合，对照直接装载的乘积（`lit/tensor/warp_frag.mim`）。
- 可信边界：swizzle 布局、描述符字段与 TMA 盒的一致性由 GPU 上的 `lit/{gpu,nvptx}/{wgmma,tma}.mim` 检查，不在 Mim 里证明；mbarrier 的相位协议在手写 kernel 里由约定保证，没有检查。

尚未进入编译路径：`lower_mma` 与 warp 拥有的区域仍固定为 TF32 m16n8k8；warpgroup 拥有的区域、生产者 warpgroup 的调度层表示都还没有。

## warpgroup 层（2026-11-13）

层次：grid ─ cluster ─ work-group ─ **warpgroup** ─ warp ─ lane。warpgroup 是一个真正的作用域：`GroupLevels#Warpgroup` 进入 `WarpgroupLevels`，
后者只提供 `Red`（面板仍铺满整个 work-group 的 tile、由所有 work-item 填充）与 `Wgmma`。参考语义仍是循环，正确性论证不变。
值的表示不需要新东西：warpgroup 的累加器就是四个 warp 各一个 `C nt`，这也是以后 warpgroup 拥有区域（FA3 的消费者）的基础。

未做：MN-major 操作数（行主序 B、attention 的 V），需要把 n 切成 64 宽的 atom、k 作行的面板布局；
TMA + mbarrier 作为流水归约层的另一种填充方式（生产者 warpgroup）；cluster 层（TMA multicast、DSMEM 归约）。
