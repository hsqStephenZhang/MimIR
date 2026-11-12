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
读 pack 时 MimIR 的规范化会直接代入 pack 体，所以剩下的读只可能读 `gpu.lanes`、`gpu.warp_mma` 或 `tensor.warp_keep`。
`warp_keep` 是恒等，标记"共享算好的值"：读别的 lane 时 shuffle 该 lane 算出的结果，而不是代入定义重算；
没有加载 `gpu` 时它在规范化时消失，所以规格在 CPU 上照常求值。

降级只改变表示，不改变值，所以一个 warp 程序在 CPU 上的结果就是它在 GPU 上的结果；前提是 warp 内收敛（`gpu.outline` 已检查 warp 集体操作不在发散分支里）。

## 层次

1. **语义层（Mim，`tensor.warp`）**：warp 值、mma 分片（复用 `mma_m16n8k8_tf32` 的 `l_a/l_b/l_c` 与 `mma_ref`）、
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
- **M2**（完成）：分布降级；`lit/{gpu,nvptx}/warp_lanes.mim` 里一个 warp 用 `gpu.warp_mma` 与 `tensor.warp.row_fold` 算乘积与行最大值，
  降成一条 `mma_tf32` 与每个半行两次 `shfl_xor`，在 GPU 上对照参考结果。
- **M3**：调度层在 attention 上，对照手写 `flash_tc` 与 `attn_fa`。
- **M4**：TileFuse 自动生成。
