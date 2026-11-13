# tritonbench 算子上的 multi-level tiling（H200 NVL，2026-11-13）

`tb.py <op> key=value…` 把 `<op>.mim` 模板（`@M@` 等占位符）实例化，编译到 nvptx，用 PyTorch 生成输入并校验结果，
用 nsys 统计每次调用的 kernel 时间；`tb_driver.cpp` 是通用驱动。用 tritonbench 的 venv 运行：

```
~/workspaces/tritonbench/.venv/bin/python tb.py softmax M=4096 N=4096 R=1 I=8 P=0
```

驱动连续调用 5 次，输入可留在 L2；表中 torch.compile 用同样“不清 L2、连续重复”的口径测（CUDA event，200 次平均），
括号里是 tritonbench `do_bench`（每次清 L2）的值。单位 ms。

## 访存密集型

| 算子 | 形状 | 调度 | 修复前 | 修复后 | torch.compile |
|---|---|---|---:|---:|---:|
| softmax | 4096×4096 F16 | 一行一个 work-group，item 8 元素 | 0.297（item 32） / 0.106 | **0.0405** | 0.0426（0.0445） |
| rms_norm | 2048×4096 F16 | 同上 | 0.0317（item 4） | **0.0102** | 0.0131（0.0159） |
| layer_norm | 4096×2048 F16 | 同上 | 0.0279（item 4） | **0.0104** | 0.0168（0.0194） |
| swiglu | 8192×4096 F16 | 4×1024 work-group，item 8 元素 | 0.104（item 4） | **0.0474** | 0.0513（0.0550） |

## GEMM + epilogue

| 算子 | (M, N, K) | 调度 | MimIR | aten | inductor / pt2 Triton |
|---|---|---|---:|---:|---:|
| addmm（bias，F16 输出） | (20120, 512, 1536) | 128×128，`wg_red_tma` 3 级，cap 尾块 | 0.0805 | 0.0608 | 0.0772 |
| addmm | (19735, 512, 1536) | 同上 | 0.0798 | 0.0592 | 0.0770 |
| linear + gelu(tanh) | (8192, 4096, 1024) | 128×128，`wg_red_tma` 3 级 | 0.166 | 0.160 | 0.163 |
| linear + gelu(tanh) | (8192, 8192, 2048) | 同上 | 0.552 | 0.529 | 0.535 |

纯 GEMM（F32 输出）同时受益于成对写回：4096³ 0.2545 → 0.225，8192³ 2.18 → 2.02（cuBLAS 0.204 / 1.82，约 90%）。

## 发现并修复的不足

1. **wgmma 片段不支持非恒等 epilogue。**
   `bias + a·b`、转 F16、gelu 落到通用 `lower_map_reduce`，类型错误；报错位置还指向 `gpu.mim` 的无关声明。
   现在 `store_epilogue` 把逐点 `map_reduce_post` 作用在片段上，其它输入沿写回的内存链逐元素读取，epilogue 本身是纯函数。
   - 第一次实现把读放在每元素的 lam 里，`fresh_mem` 让整个 fold 被复制了 256 份；
   - bias 是 region 输入的 `slice_box` 视图，每个线程 malloc 一份拷贝（`STACK:512`）；改为用 `panel::tensor_read` 穿过视图直接读源张量。
2. **片段逐元素 16 位写回。** 同一行相邻两列成对写（F16 为 `st.global.b32`，F32 为 64 位），addmm 0.100 → 0.081。
3. **绑定层不支持尾块。** M=20120 这类真实形状报 “a bound tensor.tiled level requires literal extents”。
   region 用 TMA 填充时，work-group 的最后一个 tile 按整块计算（TMA 越界补零），写回按 `origin + row < M` 跳过越界元素。
   分支的参数里带上要写的值，否则计算这对值的 fold 会被复制进每个分支（动态 shared 一度变成 3 MB）。
4. **逐元素路径的全局访存是 16 位标量。** 四个原因叠加，缺一不可：
   - 仿射映射的坐标收窄用 `conv.u`（`trunc`），`zext(trunc x)` 让 LLVM 证明不了相邻元素连续。
     映射输出按语义落在 `sout` 内，改为 nuw 的 `core.idx`，ll 后端生成 `trunc nuw`。
   - kernel 参数是一个指针数组，LLVM 不知道它们互不重叠，不敢把后面的读提到写之前。
     每个捕获的指针都是 outline 单独分配的设备内存，DeviceEmitter 给追溯到不同参数的访存加上不同的 `!alias.scope` / `!noalias`。
   - 参数指针的对齐未知。`llvm.assume` 的 align 操作数包会在指针下沉时被丢掉（InstCombine 还会把布尔形式改写成操作数包），
     所以改为在转成通用地址时插 `llvm.ptrmask(p, -16)`，对齐成为值本身的属性。
   - libdevice 函数（`__nv_expf` 等）的声明没有内存属性，调用把访存链切断；另外 `nvvm-reflect` 排在 O2 之后、O2 的展开阈值偏小。
     现在声明为 `nounwind willreturn memory(none)`，device 流水线改为 `nvvm-reflect,default<O3>`。
5. **行归约的跨 item 合并逐层过 shared memory。** 部分结果由 F32 组成、item 数为 2 的幂且是 32 的倍数时，
   先在 warp 内按分量做 5 轮 `shfl_xor`，shared 树只合并每个 warp 的一个结果（softmax 12 → 6 次 barrier）。

## 还没解决

- 调度对每个 item 的元素数很敏感（softmax item 32：0.297，item 8：0.0405），目前完全手选，没有代价模型。
- softmax 两遍都读全局的 x（L2 命中掩盖了代价）；行常驻寄存器还表达不了。
- 尾块的写回屏蔽和 TMA 越界补零是 lowering 的约定，相位类型里没有建模，`check_sync` 也不检查。

## 第二轮（2026-11-13）

### 类型层面

- **不同秩输入的 `map_reduce` 可以推断了。**
  `«i: nis; Tensor (Ris#i) (Sis#i) (Tis#i)»` 用下标变量 `i` 实例化时，按符号下标从异构元组 `((M, N), M)` 里取出的元素只有并集类型 `«2; Nat» ∪ Nat`。
  原先这会因为“tuple element is not assignable”和“expected a shape”被拒绝。现在的处理：
  - `World::component` 对依赖数组代入具体下标；
  - `check_tuple` 对按数组或 pack 自身下标取出的元素不在符号层面判错，等下标成为字面量、重新构造元组时再检查；
  - 形状检查接受各分支都是形状类型的并集。

  `softmax`、`rms_norm`、`layer_norm` 的模板因此去掉了显式类型参数；`lit/tensor/map_reduce_mixed_rank.mim` 在 CPU 上执行验证。
- **cross_entropy 用 `gather` 表达。** 标签先转成 `Idx V` 类型的张量，越界在类型里就排除了，再用 `gather` 取出 `x[i, t[i]]`；
  log-sum-exp 是一遍 online `(max, sum)` fold，用 `group_red_merge` 显式给出合并函数（`merge_with` 包裹的根推不出 `root_acc`）。

### softmax 与 cross_entropy

- **一个 work-group 可以跑超过 1024 个 item。** outline 把块大小上限定为 1024；item 更多（必须是 1024 的整数倍）时，
  线程 t 依次处理 item `t, t+1024, …`，同一 warp 的 lane 在每一轮都是相邻的 item，warp 内 shuffle 合并仍然成立。
  这样相邻线程拿相邻的小块（块循环分布），访存可以合并。
- **device 的 LLVM 流水线加了 `-unroll-threshold=600`。** 每个 item 16 个元素时循环只展开了一半（`v2.b16`），完全展开后变成 16 B 向量访存。

单位 ms；对照为同口径（不清 L2）的 torch。

| 算子 | 形状 | MimIR | eager | torch.compile |
|---|---|---:|---:|---:|
| softmax | 4096×1024 | 0.0108 | 0.0110 | 0.0132 |
| softmax | 4096×2048 | 0.0200 | 0.0234 | 0.0314 |
| softmax | 4096×4096 | 0.0385 | 0.0623 | 0.0426–0.0551 |
| softmax | 4096×8192（item 16） | 0.0658 | 0.1438 | 0.1013 |
| softmax | 4096×16384（item 16） | 0.1285 | 0.1360 | 0.1987 |
| cross_entropy | 16384×4096 F32 | 0.1176 | 0.2118 | 0.1331 |
| cross_entropy | 16384×16384 | 0.3667 | 0.6739 | 0.4069 |
| cross_entropy | 16384×65536（item 16，16 轮） | **1.168**（之前 3.03） | 4.213 | 2.037 |

cross_entropy 输出每行的 loss（`reduction="none"`），用 3 个 kernel：标签转换、行归约、最后的逐行映射。

### 旧测试

- `profile.mim`、`rules/grdeep.mim`、`tensor/group_default.mim`：`submodules/fe` 检出的版本比仓库记录的旧 19 个提交，
  profile 输出没有反引号。更新子模块后通过。
- `tensor/reduction_tiles_exec.mim`（泄漏）：k 为运行时值时，尾块的 epilogue 用了结果缓冲刚分配完时的 mem。
  AddMem 先改写参数（mem 占位在这时解析），再改写 callee，而 fold 的结果作为初值放在 callee 里，链就分叉了，
  逃逸分析于是看不到最后一次使用。现在初值来自一次带 mem 的调用时，结果缓冲从这次调用之后的 mem 开始分配。
- `tensor/group_rules_exec.mim`（泄漏）：`reference-tiles=1` 的参考 `tile_fold_impl` 是值语义循环，每一步把一块新分配的累加器作为循环参数传给下一轮。
  FreeLocals 只释放分配点本身，流进 continuation 参数的分配一律当作逃逸，所以每轮都泄漏一块。
  现在 FreeLocals 认识“受管”的指针参数，条件是：
  - 每条回边传入的都是循环体内新分配、只经 `lea`/`load`/`store` 访问、只传给这次跳转的缓冲；
  - 入口只有一个；
  - 参数自身只经 `lea`/`load`/`store` 使用。

  这样的参数在它不再存活的地方被释放，并与入口值比较：`free(acc == init ? null : acc)`。
  入口值（这里是两个循环共用的初值）永远归调用者，不会被释放。
  `mem/free_locals.mim` 增加了 `carried` 用例，测试在开启 ASan 泄漏检测时通过。
- `tensor/fuse_yield.mim`：`rejected` 的生产者被算了两遍。原因是 `compile.scalarize` 改写参数为函数的公理应用（`tiled … p body`）时，
  用一个独立的 `Keep` 改写器处理 callee，把共享的 `p` 拷贝成了组合子各不相同的两份，hash-consing 合不起来，
  于是 `outer` 那份被 `fuse_tensor` 融合重算。
  - 直接修 scalarize 会改变后面 SEO 和内联的行为；
  - 现在由 `fuse_tensor` 把 α 等价的拷贝认作同一个生产者，前提是其中一份反正要物化；两份都会被融合时（`tile_reduce_fused` 里重算 `exp`）保持原样。

  `halo`（原先被 `rejected` 的断言挡住）期望的计数是 clang -O3 消掉相邻 tile 重复的 halo 元素之后的值。
  这个 CSE 失败的原因是两次读 `x[4]` 之间夹着每个 tile 的 over-box 缓冲的 `malloc`：LLVM 推断 `malloc` 可能写 `errno`，
  而 IR 没有类型别名信息，排除不了 `x` 指向 `errno`。生成的代码从不读 `errno`，所以 ll 后端把 `malloc` 声明为只访问不可见内存，
  `halo` 回到 10（budget 模式 17）。
- 本轮还修了新出现的一个回归：最早的合并规则把两份都会融合的拷贝也合并了，导致 `tile_reduce_fused` 报类型错误。

## 第三轮：参照 CUTLASS / ThunderKittens 收紧层级类型（2026-11-13）

CuTe 的 `MMA_Atom` 把指令的操作数类型和形状编进 trait，ThunderKittens 的 `warpgroup::mma` 用 `static_assert` 检查 M/N/K 与 16 位操作数；
我们原先把这些检查推迟到 `TileFuse`（警告 “unrealized tensor.tile_req” 后静默退化）或 `lower_to_mem`（报错位置常常不对）。

- 层级链的第三个索引从 `Bool` 改成**操作数种类的类型** `NoMma` / `Tf32Mma` / `HalfMma`，由 `root_contraction` 从根的输入类型算出。
  `Mma` / `Wgmma` 层带一个该类型的值，所以层级链的结构依赖这个索引：F32 收缩配 `wg_mma`、F16 收缩配 `(16, 8, 8)` 的 `warp_mma`
  都在 `tile_req` 的调用处报类型错误（`error/tile_mma_operands.mim`）。
  只换索引而不让结构依赖它时，两个实例会被当成同一类型，错误要到 `RWPhase` 重建时才以无位置的 “cannot coerce” 出现。
- `warp_mma` 的种类由形状决定（`k = 8` 为 TF32，`k = 16` 为 16 位），因此无需隐式参数推断。
- 新公理 `tensor.require (cond, value, msg)`：条件归约为真时就是 `value`，为假时在构造处报 `msg`。用于构造器上局部可判定的约束：
  - `wg_mma`：`(64, 8 j, 16)` 且 `8 j ≤ 256`（`error/tile_wgmma_shape.mim`）；
  - `warp_mma`：`(16, 8, 8)` 或 `(16, 8, 16)`；
  - `serial_cluster` / `serial_cluster_split`：2 到 8 个 work-group（`error/tile_cluster_size.mim`）；
  - `serial_persistent`：至少一个 work-group；`wg_red_ws`：`pingpong` 为 0 或 1。
- 跨层约束也提前到了 elaboration，做法对应 CuTe 把 tile 形状放进类型：
  - `GroupLevels (n, Acc, Mma, group)` 以 work-group 的 tile 为索引，`serial_group (sizes, …)` / `group_tile_req` 要求下层是 `GroupLevels (…, sizes)`；
  - `WarpgroupLevels (n, Acc, Mma, group, tile)` 再以 warpgroup 的 tile 为索引，`group_warpgroup (sizes, …)` 要求 `tile = sizes`，
    而 `wg_mma (m, n, k)` 的类型是 `WarpgroupLevels (…, (m, n, 0))`，所以 warpgroup tile 必须是 64 行、N 等于 wgmma 的 N
    （`error/tile_wgmma_rows.mim`、`error/tile_wgmma_n.mim`）；
  - 索引只有出现在结构里才会被区分和推断，所以 `Wgmma` / `RedWs` 各带一个 `ScopeTile tile` / `ScopeTile group` 的证据；
  - `wg_red_ws` 用两个索引算出计算 warpgroup 的个数，`pingpong 1` 要求恰好两个（`error/tile_pingpong_consumers.mim`）；
  - `serial_persistent` 用 `match` 检查下一层不是 `Cluster`（`error/tile_persistent_cluster.mim`）。
  `lower_to_mem` 里对应的检查保留，作为非字面量链的兜底。
- 生成代码不变：addmm (20120, 512, 1536) 0.0796 ms，linear_gelu (8192, 4096, 1024) 0.1662 ms，softmax 4096² 0.0383 ms，cross_entropy 16384×65536 1.169 ms。

### 与参考实现逐条对照

| 参考里的静态检查 | 出处 | 我们的对应 | 时机 |
|---|---|---|---|
| MMA atom 的操作数类型（F16/BF16 vs TF32） | CuTe `MMA_Traits<SM90_64xNx16_F32F16F16_SS>`、TK `warpgroup::mma` 的 16 位断言 | 层级链索引 `NoMma`/`Tf32Mma`/`HalfMma` | elaboration（类型） |
| MMA 指令形状 | CuTe GMMA atom 的 `Shape_MNK`、TK `static_assert(M == 64 …)` | `wg_mma`/`warp_mma` 的 `require` | elaboration |
| 指令 tile 与所属 tile 匹配 / 整除 | TK `warpgroup.cuh` 的 `static_assert(D::width == N)` 等，CuTe `tile_to_shape` 的整除 | `WarpgroupLevels` 以 warpgroup tile 为索引（必须相等），`WarpLevels` 以 warp tile 为索引（`warp_mma` 要求整除，`error/tile_warp_mma_divides.mim`） | elaboration |
| 消费者 warpgroup 个数 | CUTLASS `sm90_gemm_tma_warpspecialized_pingpong.hpp` 固定 `NumMmaWarpGroups = 2` | `GroupLevels` / `WarpgroupLevels` 的 work-group tile 索引算出个数 | elaboration |
| cluster 大小 | CUDA 的可移植上限 8；CUTLASS `cluster_launch.hpp` 设 `NonPortableClusterSizeAllowed` 后放宽到 `MaxClusterSize = 32`，我们不设该属性 | `serial_cluster` 的 `require` | elaboration |
| tile 调度参数合法（tail、向量层最内） | CuTe 的形状都是编译期参数 | 构造器上的 `require` / `match`（`error/tile_tail.mim`、`error/tile_vec_innermost.mim`） | elaboration |

仍然在 `TileFuse` 晚检的，都依赖根运算本身而不只依赖层级值：
- `promote` 的位不超过输入个数，以及并行层不切归约循环：需要根的输入个数和 `ro / rn` 划分，而层级只以总循环数 `n` 为索引；
- 操作数 A 行主序 / K 主序：对应 TK 中 `st` 的 layout 类型，但在我们这里，layout 是 promotion 之后才确定的；
- lane 归约的 lane 数（2 的幂且 ≤ 32）：需要归约循环的长度；
- warp 内 MMA 的 k 块整除：`WarpLevels` 的索引只能沿链原样传递（nom 不支持非一致递归），`warp_red` 改变的 k 块进不了索引。
