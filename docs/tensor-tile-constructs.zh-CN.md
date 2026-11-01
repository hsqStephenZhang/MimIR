# Tensor-level tiling constructs（Task 7）

## 4. Destination tiled lowering（Task 8）

默认 `lower_tensor` 保留 tiled，`lower_to_mem` 顺序遍历 tile grid；
每维 full extent 与 tail extent 分开生成 callback，body 实例的 ext 均为 literal，
singleton axes 仍按 logical rank 传递。outer destination 只分配一次。
root mrp 的 map_out 加 composed origin，post read maps 减 origin 后使用原 map；
nested tiled 使用同一 destination 并累加 origin。其他 root 先算 tile buffer，再 copy
owned extent，warning 明确报告该 fallback，不把临时结果当作 zero-copy。

`-X tensor:reference-tiles=1` 保留原 tiled_impl/generate 路径。
tile_constructs_exec、nested_stage_exec、pgemm_exec 都分别编译执行两个路径并对照同一
scalar reference。tiled_destination 检查一次 grid iteration 的 root mrp 不在 per-element
generate 中，直接写 whole destination，没有 tileCopy。
本阶段 tile_fold/stage 仍走原 `_impl`，其 accumulator/scratch 优化是后续独立项目。

### 4.1 Ordered destination fold

tensor accumulator 的 tile_fold 在 loop 前分配一个 buffer 并复制 init，不能修改 caller
传入的 init。每次 step 携带 `[mem, acc_buffer]`，offset 为 iteration × ks。
root mrp 或 nested tiled 在本地证明通过后写同一 accumulator，不每步复制。
证明要求 acc 只出现在 outs，输入/combiner/post/map 无额外读；write map 为 leading
parallel coordinates、parallel extents 等于 local output extents，且 init map 与加 origin
后的 write coordinates 相同。这同时排除非单射 write 和读取其他 cell 的 alias hazard，
不能把“一个 outs use”误当作 pointwise read-before-write 证明。
unit-stride slice 的 affine semiop.mul(x,1) 仅在证明中消去，不改生产 normalizer。

未证明时整个 step 先算到临时 buffer，然后复制回 acc，并 warning 报告。
fold_destination 检查 linear step 内无 buffer.copy；tile_constructs_exec 对额外 acc input
和所有 cells 读取 acc#0 的两个非线性例子验证 fallback，两个 pipeline 数值相同。
pgemm 的完整 K-block fold 均通过证明，没有每 step 的 accumulator copy。
纯 scalar fold 保留原 ordered `_impl`；reference-tiles=1 保留全部旧 reference folds，
fresh_before/fresh_after 显式选择该路径继续测试 pinned callbacks 的 late memory threading。

### 4.2 Dependency-scoped stage realization

默认 stage 在 lower_to_mem 才展开其 reindexing 为一个 allocating fill loop。
进入 tiled 前收集 body 捕获的 stage：只有其全部 free Vars 已是 body 的 captures 时
才在该层 materialize，不能提升依赖 tile off/ext 或内层 fold 参数的 stage。
产生的 buffer 作为显式运行时 operand 传给 tile loop，per-tile rewrite 的 scoped map
复用这个 operand，不再为每个 row reader 重新生成 packing。
pgemm 的 B stage 因而在每个 column/K block 下、row tile loop 上方 realization 一次。
stage_destination 有两次 buffer.alloc：整个结果和共享 stage，无每 tile allocation。
stage_barrier 分别检查默认 stage 边界和 reference generate 边界。

`reference-tiles=1` 同时选择旧 tiled/fold/stage `_impl`，布局/padding 与 singleton tests
在两条 pipeline 均执行验证。当前保守地交由已有 mem.free_locals 管理不逃逸 allocations；
尚未显式实现 enclosing-loop scratch free 或同 extent 下跨 iterations 的 buffer reuse。
不能在有其他 scope readers 时提前 free，persistent stage/resource identity 也未实现。

## 0. Generic shape Rules

Rule 定义期仍检查 guard 的 Bool 类型，以及两侧已经闭合的类型不一致。
两侧类型依赖 Rule 元变量时，不能证明相等不再立即拒绝；每次 instantiation 的
replacement type check 必须成功，否则该 candidate 为 Unknown，不发布，继续寻找其他规则。
定义位置会发出 deferred type-check warning；实例化 type mismatch 默认以 warning 报告，
不需要 `-V`。`-X compile:report-unknown` 也报告其他 Unknown candidates。
这不是 cast 或 Hole 推断，也不允许非法 RHS 经配置成功后绕过最终检查。

开放 mutable Pi 的类型匹配不再要求模板与源指针相等：先绑定其中的 Rule captures，
再在隔离 World 中代入并只读 alpha-check。没有在 source 构造投影或修改类型。
`grshape` / `_g` 以同一规则匹配两种 shape 和动态 offsets；错误 shape 的优先规则被拒绝，
后续合法规则仍改写。`grwtype` / `_g` 从定义期 gap 翻为目标检查：
staged box 求值后才能确定的 RHS shape 仍受最终 type check 约束。

目前 generic 指 shape/offset 的 Nat 元变量；配置中含 `T: *` 的高 universe Rule list
仍需要显式适配配置的 universe，不等同于自动 universe polymorphism。

`grimmut` / `grpack` 在两个 frontends 检查 packed GEMM 所需的两项 canonicalization：
开放 immutable Arr/Pi 和开放 mutable 类型一样，在 captures 齐全后于隔离 World 内
instantiate-and-alpha-check。比如源 `«3, 4; Nat»` 应匹配模板 `«M+1, N; Nat»`
在 M=2、N=4 的实例，不能要求源的 literal 3 结构上还是 nat.add App。
这不是允许任意 term 用 alpha-check 匹配，入口仍仅限 type-level Def；最终 replacement
type check 保留。

Tuple pattern 可以匹配同长度的 immutable homogeneous Pack；`(M,K,N)` 对
canonical `‹3; 4›` 把三个 captures 都绑定成 4。每个 operand pattern 均必须匹配
同一 Pack body，长度及整体类型仍检查；重复元变量 `(M,M,N)` 不会接受 `(4,5,6)`。
非 homogeneous Tuple 沿原路径匹配，mutable Pack 不因此获得结构匹配权限。
这使 8³ 和 1×K×1 的 dimension metadata 与非方阵使用同一 generic rule，
没有针对特定 literal shape 的第二条规则。

## 0.1 Operand placement 的 dominance

`schedule_index.mim` 不依赖 tensor：parallel body 把 I32 counter 窄化为 `Idx 3`，
同一 index 用于初值读取、inner loop 的读取和 fold 后 continuation 的写回。
master 的 `Scheduler::smart` 只以 uses 的 structural `late` 为上界，各 use 独立上移，
写回 GEP 被放到 outer head，而 trunc 留在 outer body，LLVM verifier 报 operand 不支配 user。
修复取自 `feature/default-sched` 的 `2eeda4cc2d`：`smart` 的上界改为 uses 实际 `smart`
placement 的 lca，与那条分支保持同一实现。

## 1. Tensor accumulator

`map_reduce_post` 增加推断的 `Init` group，原 scalar init 调用不变。
`outs shape map_init value` 提供显式 read map，`outs_default shape value`
使用该 op 的 `map_out`。descriptor 的 shape 保留 logical rank，不从 singleton-collapsed
数组类型推断。scalar 与 descriptor 在 lowering 验证，非法 init 报错。
tensor data-use schema 包含 outs，fuse 的同 rank slice/read-through 可以合成 `map_init`；
暂不融合需要改变 accumulator logical rank 的 read。

经典、vectorized、blocked、redvec 的既有 seed 点统一调用
`init_cell: Cn [mem.M 0, «Rn; I32», Cn [mem.M 0, To]]`。
tensor.lower_to_mem 只 bufferize descriptor；btensor.lower_map_reduce 创建读回调。
outs buffer 作为运行时参数在进入 nest 前求值，不能只藏在逐 cell callback 的 capture 中：
否则 producer 会在每个 cell 重跑，且被共享的 allocation 会覆盖先前结果。
scalar 使用常量 callback，FC/redvec、conv、attention 的 scalar LLVM 去掉 ID 后仍完全相同。

`outs_exec` 检查 split fold、仅最后 block 的 I64→I32 post、空 reduction、
非 identity init map，以及 classic/vectorized/blocked/redvec。
现有 `tensor.lower_map_reduce` 保留 value-level counting-loop 参考路径，并在 reduction 前读初值；
没有另造 buffer loop nest。一般 `map_out` 不要求可逆，故未改成通过输出坐标反求循环坐标的
generate 实现；新 tiled/stage 的 generate 参考语义另列于下一节。

## 2. tiled / tile_fold / stage 的参考语义

`tiled @(T, r) (s, ts) body` 拼合互不重叠的 owned tiles。
body 的参数是全局 `off` 和该 tile 的实际 `ext`，返回 `«ext; T»`。
尾块按每维 remainder 剥离，tile 大于 extent 时缩到 extent；singleton collapse
不改变 logical rank。零 extent 不调用 body，空 tensor 的 LLVM ABI 为 void。
静态零 tile size 报错。当前 generate 参考实现会为**每个输出元素重算整个 tile**，
不是一次 tile realization，也没有分配/性能保证。

`tile_fold (n, ks) init step` 依次以 offsets `0, ks, …, (n-1)*ks`
调用 `step(acc, off)`，严格维持 Acc 类型及先后顺序，不重排非结合 combiner。
`n = 0` 直接返回 init；非空 fold 的 ks 必须正数。只包含完整 blocks，
最后的 reduction block 及最终 epilogue 由调用者在 fold 后显式构造。

`stage @(T, r, k) s (inner_dims_pos, inner_tiles, outer_perm, pad) scope input`
按 linalg.pack 的约定重索引：先把选定维度 ceil-div，按 outer_perm 的
**输出轴顺序**排列，再追加 inner_tiles。inner_dims_pos 为互异的原始轴号，
outer_perm 必须双射；这与 tensor.transpose 的 input→output 约定不同。
越界 lanes 返回 pad，不执行源 load；scope 目前仅保留语法位置，不映射存储地址空间。
lower_tensor 展开成 generate，fuse_tensor 不穿过 generate 合成读取；
identity stage 也不由 normalizer 消除。`stage_barrier` 检查该边界。

参考 callbacks 可以产生 tensor 分配，lower_to_mem 把其局部 fresh-memory 请求包装成
返回式 CPS 区域。第一次 `mem.add_mem` 前请求已经存在，不是 cps.conv 新增分配：
`tile_fold_impl` 的 nullary `run` 包含 `next → mem.fresh(0, fresh_mem)`，
`tiled_impl` 的 tail-dispatch `choose` 也包含 slice buffer 的同类请求。
两者通过 `cps.cps2ds_dep` 被调用；AddMem::analyze 把该 axiom 的回调以及其传递可达
continuations 加入 preserved_，rewrite_mut_Lam 在 preserving_ 模式原样复制它们。
因此 next 仍为 `Cn []`，不能在那里注入 memory，fresh 也没有进入普通 resolve 分支。
generate 的 element callback 经 btensor 的 axiom-pinned ABI 也存在同样限制。

CPS 转换及 affine loop 展开移除这些调用边界后，`mem.resolve_fresh` 仅在仍有
`mem.fresh` 时调用现有线程化逻辑。放在 affine.lower_for 之前会改变循环 callback 的
ABI 而使调用不匹配；第一次 AddMem 直接 unpin 同样破坏 cps2ds 的 dependent type groups。
前端分支 e76c4d05f9 / e9d856fb28 / 9d2bf7f6a7 优化的是普通 resolve 分支中
tt continuation 的 splice、递归判断及每次请求的 binding scope；
它们不改变 preserving_ 的入口，不能解决上述 pinned 区域。
本实现没有搬用其不适用的旧 Rewriter mapping 逻辑；若移植该优化，仍必须保持 scoped binding。
没有剩余请求的程序不再改写，既有 GPU pipeline 保持不变。
`fresh_before` 正向断言第一次 AddMem 的残留请求；`fresh_after` 断言 ABI 展开后请求消失；
`tile_constructs_exec` 验证 repeated fold/tail iterations 的数值，不把“无 fresh”当数值正确的代理。
目前这些参考构造只验证 CPU LLVM，GPU scope 与 address spaces 不在此实现内。
helper ABI 的识别只沿实际 metadata callbacks，不沿输入 producer 或 Var binder 回边。

`tile_constructs_exec` 对照 C++ 标量参考检查 divisible/tailed/singleton/empty tiled、
一轴及双轴 pack、padding、标量非结合 fold 和 tensor accumulator fold。
该测试也覆盖空 tensor 的 erased ABI；非法 tile、block size、布局有正向诊断断言。

§5.4 destination-passing 尚未实现：需要 outer destination 的 logical shape/origin、
每次 iteration 的 memory/accumulator、scratch scope/lifetime、outs 的线性使用检查，
以及不能原地写时的显式 copy/report。当前 fold 的中间 heap buffers 可能保留到调用结束
之后；不能把参考实现当作 scratch 生命周期或 persistence 契约。

### 2.1 Nested closure 的 fresh-memory memo

`nested_stage_exec` 的最小结构为 outer tiled → stage(slice(input, off, ext))
→ inner tiled → slice(stage, row, size)，不需要 tile_fold、outs、combiner 或 captured access map。
单纯 nested tiled 读取输入、单个 tiled 内的 stage、单个 tiled 内的 mrp/slice、captured combiner
和 captured access map 均不是这次 assertion 的充分条件。
当 inner row 和 outer column 有尾块，原 lower_to_mem 留下 stage allocation 的 fresh_mem Var。
失败的是闭合 external `stage_share` 经 RWPhase::rewrite_external 统一 externalize，
不是 lowering 把某个 local Lam 主动提升为 external；相同结构不经 Rule 也失败。

只给 allocating App 一个 per-Lam fresh_memo_ 不够：它的 users、argument tuples、
slice 组合仍进入 Rewriter 的全局映射，并可把另一个 closure 中的 fresh_mem Var 带回来。
每个 Lam 现在打开 Rewriter scope，新建 fresh_mem Var 登记为该 scope 的 new_var；
依赖它的映射只在该 Lam 内有效，其他不依赖该 Var 的映射仍可共享。
结束时销毁这些临时映射，不改变 ordinary closures 的 environment 或 external ABI，
也不新增第二套 loop builder。测试执行两次、每次改变输入并检查全部 tail cells。

### 2.2 AddMem 的 continuation-local memory memo

`mem.add_mem` 和 `mem.resolve_fresh` 共用 AddMem，均在每个普通 Lam 的 body
改写前打开 Rewriter scope，并登记该 Lam 的 memory 参数所属的 Var。
allocation、projection、argument tuple 等结果只要依赖这个 Var，其 memo 就留在该 scope；
离开 Lam 后删除，不把另一 continuation 的 memory token 带回来。
不依赖该 Var 的映射仍留在 root，Lam stub 在打开 scope 前登记以保留递归与闭合共享。
pinned ABI 的 preserving_ 路径不注入 memory，不改变已有 GPU callbacks 的签名。

`add_mem_shared` 的两个 continuations 都使用同一 hash-consed
`mem.alloc (I64,0) ⊥:(mem.M 0)`，`resolve_shared` 再通过 fresh 接收 callbacks 测试
第二次 pass。原实现两个文件都在 externalize 时 assertion；新实现各自在本地 memory
上 allocation，检查输出 token，且没有残留 fresh。不是针对 alloc 的特判。
这也修复同 module hand/rule packed GEMM 的 late-pass allocation/projection memo 泄漏。

### 2.3 AddMem 兼容性检查（Task 7f）

修复前后用同一 Debug build 配置及默认 `-p opt … -p ll` 生成
`fc_exec`、`attn_exec`、`buffer/exec_conv` 的 LLVM；仅把 `_数字` 生成 ID
统一成 `_ID` 后，三个文件完全一致，没有 placement、instruction 或 ABI 差异。
`exec_conv_run` 对该 conv module 做实际执行并检查 exit code 63。
完整 CTest 235/235，FULL lit 609 passed / 9 unsupported；
包含已有 GPU/NVPTX 编译路径，unsupported 项不代表 GPU 硬件执行已经验证。
原始 LLVM、去 ID diff 和完整日志保存在 `build/task7f-before`、
`build/task7f-after`、`build/task7f-item1-{ctest,lit}.log`。

## 3. Generic packed GEMM（Task 7f）

`lit/tensor/pgemm.mim` 的同一 Rule 对 M、K、N 和 MR、KR、NR 泛化，
把 marked real product_2d 改写成 tiled(N) → tile_fold(K, outs=acc) → tiled(M)。
每个 K block 在 row tiles 外绑定 `bp = stage(slice(B,…))`，
packed read map 为 `(j/NR, k, j%NR)`；A 与 accumulator 按实际 row ext 切片。
先 fold `(K-1)/KR` 个完整 K blocks，再单独执行最后一个正长度 block。
M/N 尾块使用 actual ext，stage 的 padding lanes 不进入逻辑结果。
这里固定 I64 wrapping add/mul，保留 matched ring operators；plain GEMM 不含 bias/ReLU，
不声称浮点结合性或 reduction 重排权限。

`pgemm_exec` 在**同一个 LLVM module** 包含三个 Rule-generated 与三个直接调用
`packed` 的 hand-written externals。两种路径均与 C++ scalar reference 逐 cell 比较：

| Shape M×K×N | MR×KR×NR | Rule / hand |
| --- | --- | --- |
| 8×8×8 | 4×4×4 | 全部正确，重复两次且改变输入 |
| 13×7×11 | 4×3×4 | M/K/N tails 全部正确，重复两次且改变输入 |
| 1×7×1 | 2×3×4 | singleton-collapse ABI 与 K tail 正确 |

`pgemm` 的 IR 测试检查 stage/mrp/tiled/fold 存在且所有 marked calls 消失；
graph frontend 保留三个 marked roots，并正向检查 recipe builder 的拒绝诊断。
拒绝的是 RHS `packed` local Lam call，不是缺少一个 pure-axiom 名字；没有放宽
graph purity 合同。支持它还需要纯 closure-call recipe 或安全的 helper 展开。
执行使用 rule_app，经现有 `_impl` 参考语义 lowering。
最终 CTest 235/235、FULL lit 611 passed / 9 unsupported；
同一 C++ reference 也链接 `clang -O2` object 重新执行并通过。

### 3.1 编译时间与代码体积基线

macOS arm64，同一 Debug Mim build，默认 `-p opt` → LLVM。
命令为 `build/bin/mim -I lit/tensor -p opt lit/tensor/pgemm_exec.mim -p ll`，
在独立输出目录执行，包含全部六个 externals；先 warmup 一次，再测五次。
计时含 frontend/plugin loading 和所有 Mim phases，不含随后 clang/object 编译或运行。

| 项目 | 结果 |
| --- | --- |
| 五次 wall time（秒） | 0.643、0.640、0.657、0.663、0.662 |
| 中位数（秒） | 0.657 |
| LLVM IR | 356,110 bytes / 7,942 lines |
| clang -O0 object | 42,720 bytes；Mach-O __TEXT 39,712 bytes |
| clang -O2 object | 19,104 bytes；Mach-O __TEXT 15,920 bytes |

原始命令、计时、LLVM、objects 和 numerical executable 位于
`build/task7f-gemm`，完整 metadata 为 `metrics.json`。
这些不是运行速度或 allocation 数量的优化结果，也不是 per-function code size；
对象体积不包含 C++ reference 或 linked runtime。

### 3.2 性能 lowering 尚缺

当前 tiled 的 generate 参考实现可为每个 output cell 重算整个 tile，
语法上的外层 bp binding 尚不保证 stage 运行时 exactly-once realization。
§5.4 destination passing 需要传入 outer destination 的 shape/origin、每次 iteration
的 memory/accumulator、scratch lifetime，以及 outs 的线性 alias 检查。
不能原地写时须显式 copy/report；fold 中间 buffers 的释放与 reuse 也需实现。
因此本实验验证 generic Rule、closures、tail 数值与完整 lowering 通路，不是 packed GEMM
的性能实现；GPU storage scopes 和 fused bias/ReLU epilogue 尚未在该实验验证。
