# P5 tile-and-fuse 实验

## Task 17A：默认流水线的显式开启验收

默认仍为 OFF；本节仅对 `-X tensor:group=1` 与 `group=0` 比较。
`lit/tensor/group_default.mim` 不含 Rule、marker 或 `_compile`，
GEMM+bias+ReLU、MLP、softmax、layernorm、两层 conv 均走 `opt._default_compile`。
group=0 仍保留原来的 L1 producer/epilogue fusion，不等同于完全禁止融合。
两个模式均通过 scalar reference 与 ASan；GEMM/MLP 输入为有正负值的非整除小形状，
row normalization 为 5×7，conv 为 NCHW 两层 3×3。

小形状 profiler 共 7 个请求、3 次 Rule rewrite、4 个残余报告，bufferization 没有接收 marker。
3 个 kind 1 分别为 GEMM 和 MLP 的两层，bias/ReLU 在末 K block 执行；
2 个 row kind 2 未匹配 live consumers，按合同报告并回退；
2 个 conv kind 1 不满足 BLIS 的两 parallel axes/two inputs 前提，报告后保留原 schedule。
没有把这四个回退记录为实现成功。

### 计时设置与结果

`scripts/bench-default-groups.py` 从同一 helper 源生成无规则的 group=0/1 模块，
统一链接到 `group_default_bench.cpp`。
Apple M2 Pro，Homebrew LLVM 22.1.4，F32、单线程、`-O3 -mcpu=native`，
每项 1 warmup + 7 runs 取中位数，轮换各模式的执行顺序；时间包含返回值 allocation/free。
计时前与结束后检查均没有其它 mim/clang/cmake/ninja build。
矩阵 benchmark 先检查两种输出与 OpenBLAS 结果一致；
OpenBLAS 也是单线程，包含独立 bias/ReLU loop、MLP 中间值与输出的 allocation/free。
row/conv benchmark 在 warmup 时逐项与 scalar reference 比较。

下表单位均为 ms；OpenBLAS 列包含独立 elementwise loop。

| 工作负载 | shape | group=1 | group=0 | OpenBLAS + loop | group0/group1 |
| --- | --- | ---: | ---: | ---: | ---: |
| GEMM+bias+ReLU | 256³ | 0.924333 | 1.211625 | 0.586833 | 1.311 |
| MLP | 256³ × 2 | 1.575667 | 2.102416 | 0.910709 | 1.334 |
| GEMM+bias+ReLU | 512³ | 5.882292 | 7.862708 | 2.868250 | 1.337 |
| MLP | 512³ × 2 | 11.214250 | 15.578333 | 5.334166 | 1.389 |
| GEMM+bias+ReLU | 1024³ | 47.232875 | 65.309583 | 24.201208 | 1.383 |
| MLP | 1024³ × 2 | 92.699375 | 131.129125 | 46.035167 | 1.415 |
| GEMM+bias+ReLU | 2048³ | 358.387167 | 611.626875 | 166.904667 | 1.707 |
| MLP | 2048³ × 2 | 1038.103833 | 2298.454416 | 382.704958 | 2.214 |
| softmax | 1024×1024 | 3.518125 | 3.595916 | — | 1.022 |
| layernorm | 1024×1024 | 0.210208 | 0.200958 | — | 0.956 |
| softmax | 4096×1024 | 18.790542 | 15.727583 | — | 0.837 |
| layernorm | 4096×1024 | 1.126792 | 1.145125 | — | 1.016 |
| softmax | 1024×4096 | 18.585166 | 19.101958 | — | 1.028 |
| layernorm | 1024×4096 | 1.176917 | 1.143666 | — | 0.972 |
| conv chain | N=2、Cin=Cout=8、32²→30²→28² | 0.080917 | 0.082542 | — | 1.020 |

benchmark profiler 共 14 个请求、12 次 Rule rewrite，只有两个 conv residual warnings。
矩阵 4 个尺寸的 12 个 contraction 实现 BLIS；默认 F32 selector 为 `(MR,NR,KC,NC)=(5,32,16,256)`。
这里不是早前手写 BLIS benchmark 的 256-KC 配置，不能沿用其 94–96 GFLOPS 结论。
大形状 row normalization 没有发请求：
256-grid-point cost-sampling 上限与 32 KiB scratch cap 的组合没有找到可接受候选。
因此表中的 row/conv 差别不构成融合收益，softmax 4096×1024 的下降也照实保留。
自动 GEMM/MLP 虽比原默认路径快，仍明显落后 OpenBLAS；
本项没有修改成本模型、放宽 legality 或把回退冒充已实现的 row/conv grouping。

可重复入口为 `python3 scripts/bench-default-groups.py`；
`--build-only` 只生成可执行文件，默认入口会等待其它 compiler build 退出后才计时，
超过 5 分钟仍繁忙则报错而不输出计时。
本次原始输出位于 `build/task17-default-bench/results.csv`，
完整请求计数位于同目录 `profile.txt`。
`reduce_exec` 的 closure lowering 问题由并行会话处理，本任务没有修改该路径或默认开关。

## Task 16B：先分组，再实现 Rule

`tensor.group_req` 的共享接口先独立提交为 `97dfe96fad`。
后续 `tensor.group` 在 L1 后自动生成 kind 0/1/2 请求：
复用已有 grouping 根策略、共享 BLIS 目标 selector、自动发现合法 reverse consumers。
本阶段不改默认 pipeline、不运行组内 Rule；测试通过显式 phase pipeline 检查 marker。
详细字段、嵌套请求、成本和诊断合同见 `tensor-tile-constructs.zh-CN.md` §11。

| 用例 | 本阶段选择 | 与手写 TF-RULES / default-sched 的差异及原因 |
| --- | --- | --- |
| row reduction → column reduction | kind 0，dim=0、tile=4 | 根一致；小图完整工作集在预算内，取代手写 tile=2 |
| contraction → column reduction | contraction 保留 kind 1 | 延续 contraction boundary，不把它当普通 producer 重算 |
| 重复 reader 链 | 保留根策略及重算预算检查 | 不按 reader 静态数量代替执行次数 |
| rowsum reverse，5 行尾块 | kind 2，dim=0、tile=2，P 在 C 前 | 与显式 reverse Rule 相同，不再需要 common-root marker |
| two / swapped consumers | kind 2，按源 gid 排列 consumers | 保留一份 P；与显式 tuple 的 slot 顺序可能不同，但消费关系不变 |
| 小 cache 的 `4×1024` producer/consumer | cache=4096 时 dim=1、tile=64、scratch=3072 | 不固定 outermost；axis 0 的 whole-width 工作集超预算 |
| GEMM + epilogue | kind 1，I64 默认 `(MR,NR,KC,NC)=(5,32,8,256)` | 与抽取前 BLIS selector 相同；epilogue 留在 L1 合并后的 contraction 内 |
| 两层 MLP | 两个 kind 1，并在层间选择 row reverse | 手写结构 Rule 外增加自动的跨层请求；不将 contraction 误作普通 outer root |
| SwiGLU | 两个 kind 1，无 kind 2 | L1 将 G⊙U 融到 G 的 post；U→H 仅是 post reader，现有 reverse 需要 ordinary input，不能照搬 pre-L1 的显式选择 |
| softmax / layernorm | row helper 上生成请求，保留已有 stage barriers | 不穿过 retained-stage 分组，也不删除稳定性所需的中间值 |
| 已显式 tiled 的 halo conv 链 | 保留 tiled 及 callback-local placement | 不对已归组 region 再分组；不重复执行手写 halo Rule |
| 未显式分块的 conv 链 | contraction kind 1 | 仅请求目标结构分块，不声称 conv 几何已满足 BLIS GEMM recipe |
| 7 元素 producer、五个 width=3 windows | cache=80、budget=2：kind 0、tile=2、factor=11/7 | 自动成本包含 halo overlap；budget=1 拒绝 |
| consumer 跨 P 的 tile 轴 reduction | 不发 kind 2，日志要求 ordered fold | 不改变 reduction 次序，不以完整轴 tile 规避拒绝 |

`feature/default-sched` 的 group.cpp 选择 contraction/placement roots 和非 vector 的
outermost parallel axis；本分支没有其 schedule actions、prepack/cache 字段，
因此沿用本地 `GroupPolicy` 的边界，并显式搜索多轴工作集。
精确 boxes 优先于重算比，再选 cache 内最大候选 tile；
source/uses/count 不完整、动态 storage、自定义 chooser 和分析上限均保守拒绝。
请求携带 source debug key；未实现请求由 `report_groups` 或 lowering 发 warning，
strict 模式报错。后续组内 Rule 和默认 pipeline 集成由配套任务负责。
没有执行时间 benchmark，也不把 marker 检查当作 realization 的数值正确性证明。
提交前完整验证：CTest **243/243**；FULL lit **655 passed、9 unsupported**
（共 664，无失败或 XFAIL），包含已有 BLIS selector/execution 回归。

## Task 14B：reverse_compute_at common-root Rule

`reverse_at (P, C, dim, tile_size)` 显式选择 producer 和一个/多个 consumer，
`reverse_rule` 以冻结 `used_by` 和纯 affine box geometry 为 guard，
生成 P-first 的共享 grid `tiled_n`。
检查 P 完整覆盖、write image exact、write preimage 唯一，
以及 C read preimage exact、完整 reduction 和独占输出 box。
跨 tiled dim 的 consumer reduction 明确拒绝并要求 ordered fold；
side/post/accumulator 的间接依赖也拒绝，不交给 lowering 猜测 placement。

两种 Rule frontend 执行 rowsum、two、swapped、SwiGLU-shaped gating，
另检查 whole reader、inner placement/transpose、直接 producer epilogue，
与 scalar reference 逐元素比较。destination 和 reference-tiles 均通过；
乘法计数证明 P 不因多个 consumers 重算，未测运行时间。
八类非法请求保留 marker，并在 lower_tensor 输出拒绝原因。
支持边界和完整 proof 条件见 `tensor-tile-constructs.zh-CN.md` §10：
literal canonical MRP、最多 65536 grid points、显式 common root，
始终保留 whole P slot；SwiGLU 的独立 U 保持 whole。
提交前完整验证：CTest 242/242；FULL lit 649 passed、9 unsupported。
最终增量 build 的 wall time 为 6.93 s；未测任何执行时间。

## Task 13B：显式 halo Rule

`tensor.fuse_halo (T, rank, budget)` 将保留 image-cap witness 的 `slice_box`
改写为 `tile_of_box`；native `within_budget` 从冻结 program snapshot 累加每次
over-box 的有效 cells，重叠不去重、dilation 空隙不扣除。
budget=1 拒绝 `11/7` 的 window 重算，budget=2 接受；
`15/7`、whole reader 的 `18/7` 和未知 fold count 均拒绝。
该 opt-in Rule 可递归计算 canonical conv producer，不改变 `fuse_producer` 的 contraction roots。

`tf_halo_exec` 对两个 frontend 检查 IR，并执行 stride 1/2 + padding 1、
dilation 2、一行/不整除 tile，以及 conv → ReLU → conv；
destination 与 reference-tiles 路径均与标量实现比较。
padding 留 whole，scale 放在 zero-pad 后再按 halo 计算；
不跨越 pad 融合其内部 producer，也不改动默认 conv schedule。
box-cap 方程、纯性责任与支持边界见
`tensor-tile-constructs.zh-CN.md` §7.7。
提交前完整验证：CTest 241/241；lit 644 passed、9 unsupported，无失败或 XFAIL。
五个负载一起编译到 LLVM：whole-producer/rule halo/graph halo/reference-tiles
分别为 0.564/0.533/0.577/0.574 s；未测执行时间。

## Task 6：受限 term-LHS binder matching

Rule LHS 的 Lam 不再只按指针匹配：matcher 显式绑定源 Lam 和 Var，检查函数 type/filter，
再递归匹配 body。绑定变量必须仅作为变量出现；bound-function callee 的高阶匹配显式拒绝。
没有高阶统一、逆归一化、Hole 求解或开放 dependent Pi 推断。
RHS 的对应 binder 按 Lam preorder 与嵌套深度配对，type 必须只读 alpha-compatible；
同数、同深度的受限 nest 覆盖本实验的 jo/io。
capture 若含源 bound Var，只可在 RHS 对应 binder 内使用；否则拒绝该 match。
导入延迟到 RHS binder stub 已安装后，冻结源表达式按该映射重建，不能把源 Var 搬出 scope。
`gralpha` 在两个 frontend 验证 binder-dependent capture 和 escaping rule 后的安全 fallback。
不支持增删/重排 binder 后仍推断 capture 的 scope 对应关系。

`tf_post` / `tf_post_g` / `tf_rpost` / `tf_rpost_g` 全部翻转为目标检查。
真实 slice 版本生成 jo body 中的 bp，并将 io body 读取改成 read bp。
模型 mm 版本的 stage 虽在 RHS jo 的 let 中构造，仍打印在 io 的 mm 参数里：
它只依赖 jo 的 box，尚无 placement/lowering 合同，不能声称此模型已经执行 outer-level packing。
graph recipes 对源 nested body 的未证明纯性计算也检查保留，不能借 binder pattern 删除 effect。

## Task 6：staged witnesses

### 纯计算与源分析的边界

| 计算 | 求值方式 | 依据 |
| --- | --- | --- |
| literal permutation 对 box 的重排、整数边界运算、显式 affine/layout 描述的组合 | 普通纯函数；native 实现可注册 normalizer | 输入是显式数据，不读 graph、mutable binder 或用途覆盖 |
| 从 mutable map Lam body 提取 permutation、从源 producer 识别访问关系 | staged witness | 必须分析 matched source operand，不能分析已重建的 RHS |
| consumer/exclusivity、cover/preimage、alias 或 reduction-order 证明 | staged guard/witness | schema、覆盖和可证明性必须显式；Unknown 不触发规则 |

本轮没有把所有 box 计算都升级为 graph query。重排的数值运算由共享的
`permuted_box` helper 完成；读取 mutable Lam 或源 transpose 则由 staged callback 完成。
对于运行时 offset，可以让后续 witness 返回 literal permutation/affine 描述，
再由纯函数在 RHS 中组合现有 capture；不能把源 Var 或可执行 App 冒充 witness data。

### 注册、冻结、memo 与导入

`include/mim/pattern/witness.h` 的 `Witness` 包含最终 axiom argument group 的字段数
`num_inputs`，以及 `evaluate(const GraphView&, Defs inputs, World& result)` callback。
隐式 type/rank groups 不传给 callback，结果仍须符合实例化后的 axiom return type。
plugin 用 `Plugin::register_witnesses` 按 annex flags 注册；native client/test 也可用
`Driver::register_witness`。同一 flags 不能同时注册 ordinary normalizer。
本轮的 domain-specific axioms 放在 tensor plugin，不增加 Rule/parser 语法或 guard 输出参数。
Plugin ABI 增加 callback 字段，binary plugins 需要与 libmim 一起重新构建。

规则写普通的 witness axiom call：

```mim
let box = tensor.input_box @2 (map, (1, 4), (2, 3));
```

调用在 Rule guard/RHS 模板中发现，LHS captures 和结构对应关系将参数冻结为源 handles；
不能对应到源的计算表达式、没有 GraphView 的调用均为 Unknown。
显式 immutable literal data 可以直接作为分析输入。不会导入或重建 source map 后再分析。
两个 frontend 复用每次 phase run 的 source snapshot；nested region 仍使用 Open view，
free variables 不成为用途覆盖证明。查询用途的 callback 必须使用已有显式 data schema，
注册 witness 不提供 purity 或 exclusivity。

每个模板调用每次 match 最多执行一次；同一调用同时用于 guard 和 RHS 时共享结果。
返回 `True + value` 才可绑定；`False` 为 No_Match，`Unknown` 为 Unknown，均不发布。
memo 将模板表达式本身作为 synthetic capture key，没有新增未绑定的 Rule 元变量。
结果必须在专属 result World 中，是闭合 immutable Nat/静态 Idx literals、Tuple/静态 Pack，
并通过 declared result type 的 alpha check。错误类型、mutable/可执行结果都跳过并诊断。
callback 的 source snapshot/gid 变化是违反 native contract 的错误，不是正常的 Unknown。

GuardResult 持有 result World；graph RewritePlan 继续持有它直到 transaction 被导入或丢弃。
两个 RHS frontend 都通过**非重写 importer**复制结果，绝不再次运行 rule 或普通 normalizer
来求这个 witness。graph 的新 executable axiom calls 仍要求 pure_ops，最终 replacement
type/scope checks 不变；witness data recipe 不是放宽 RHS purity 的调用通道。
guard/RHS 中有 witness 的规则使用 occurrence-scoped publication，保守避免把局部分析
推广到共享 closure 的其他 occurrence。
普通程序中的 witness call 仍未求值，`tf_wbox` 的 ordinary 函数对此作正向检查。

### 非 identity box 实验

| 实验 | rule_app | graph_rule_app | 检查 |
| --- | --- | --- | --- |
| `tf_wbox` / `_g`，modeled reindex + 真实 slice/unary | 目标通过 | 目标通过 | mutable transpose map 将 offset `(1,4)` / extent `(2,3)` 变为 `(4,1)` / `(3,2)`；diagonal map 为 Unknown，源 IR 保留 |
| `tf_rbox` / `_g`，全部使用真实 slice/transpose/unary | 目标通过 | 目标通过 | 源 transpose 提供 permutation；guard 证明 extent `(3,2)`，RHS slice 使用 witness offset `(4,1)` |
| `grwtype`，staged extent 直接成为真实 RHS shape | 正向记录共同的定义期 gap | 同左，尚未进入 phase | `type mismatch between rule sides: lhs has type «2, 3; Nat»`，RHS shape 仍含 symbolic `box#...` |

`tensor.input_box` 支持 rank 1–64、set map Lam 和 bijective projection body；
`tensor.transpose_box` 从冻结的真实 transpose 读取 literal bijective permutation。
两者当前都要求 literal offsets/extents，只返回 box，不宣称 in-bounds、cover、alias、
recompute budget 或 exactly-once 已得到证明。broadcast、floordiv、affine arithmetic 和
动态 offset 的通用 box 还没有 callback 实现，不能由这些实验推导合法性。

完全真实的实验采用安全的 shape specialization：guard 和 RHS 的 witness 是同一调用，
guard 验证两个 extent 字段后，RHS 用已声明的 literal shape，offset 仍由 witness 提供。
这没有靠 Hole、cast 或延期检查绕过 Rule 两侧定义期 type equality。
一般 staged-dependent RHS shape 需要 type refinement/proof 或受检查的 typed recipe 扩展；
`grwtype` 保留精确诊断，不能把此项报告为通用 dynamic-shape compute_at 已实现。

两个 C++ 回归验证 guard/RHS 只执行一次、original source handles、result World ownership、
没有 GraphContext 时不执行，以及 Unknown/False/错误 type/mutable result 的拒绝。
tensor C++ 回归额外检查三维非自反 permutation，防止只在二维 transpose 上混淆方向。
以上全部是 IR/box 实验，未新增 lowering、真实 packing/persistence 或调度性能合同。

### Task 6 验证

| commit 项目 | CTest | lit rules/error/tensor |
| --- | --- | --- |
| alpha-aware term-LHS binders | 231/231 | 221 passed，4 unsupported |
| staged witnesses 与真实/模型 box 实验 | 234/234 | 226 passed，4 unsupported |
| 本文结果与剩余合同更新 | 234/234 | 226 passed，4 unsupported |

每项 commit 前运行完整两套测试，无失败或 XFAIL。初次迭代中曾触发 deep-DAG 回归和
实验 CHECK/定义期 shape 失败；已修复深 DAG 的 no-witness fast path、非重写结果导入
与 CHECK 顺序，通用 shape gap 则由 `grwtype` 正向保留，没有隐藏失败。

## 后续修复：自然 telescope

`(x o e: Nat)` 的 LHS 分别捕获字段，但 RHS 的 `(x,o,e)` 经 tuple η 归约变为
whole-telescope Var；`(a b: «4; Nat»)` 的 LHS 则可能直接捕获 whole Var，RHS 需要其字段。
Rule adapter 现在支持两个方向：完整字段在 target World 中组成 tuple，whole capture
在 target 中静态投影；若源 Tuple/Pack 已有字段，绑定直接补充其现有节点。
不制造 source Extract、Hole 或猜测 dependent-type 元变量。
guard、rule RHS 与 graph binder recipes 使用同一 derived-binding 规则；
source graph queries 不会将新建 target 投影误认为 source 节点。
`grtel` 在两个 frontend 覆盖 homogeneous scalar、array-valued 与 mixed telescope，
不需要 unused/salt 字段。真正不由 LHS 绑定的 RHS/guard 元变量在 phase 配置时，
按 Rule 定义位置显式报错（`grunbound`），不再仅以 debug 日志静默跳过。
当时 fresh LHS lambda 的 alpha-matching 缺口独立保留；其词法元变量是已定义的，
不能将不匹配误报成非法 Rule。此项现由 Task 6 的受限 binder matching 覆盖。

## 后续修复：共享 region 的发布模式

此前所有 nested rewrite 都克隆 operand occurrence，纯等式在 fixed point 下也留下多个
相同 mutable Lam。现在检查 guard/RHS 是否包含 staged graph predicates 或 witness 调用；
没有 graph facts 的规则在选中 enclosing function 内按 Lam value-wide 发布，共享的
region binder 只重建一次，所有该值的 occurrence 复用其结果。
有 graph facts 的规则仍按 occurrence path 发布，不能把局部用途判断推广到其他位置。
检查会遍历模板的嵌套 lambda；超出分析预算时保守使用 occurrence 模式。
`grshare` 验证纯规则 ff/tt 都只留下一个 body，graph-fact 规则 ff 只改一个 occurrence，
tt 可分别改写但保持两个 binder。`grregion` 的 occurrence 隔离用真实 graph-fact guard
继续覆盖。这个复用限定于同一 enclosing function，不是跨函数 region/resource cache。

## 后续修复：binder debug metadata

graph rebuild 手工映射 scope/region Var，绕过 Rewriter 的常规 `set(dbg_key)` 路径，
导致原参数 x 和 offset o 被打印成匿名名字。
现在创建 fresh binder 时同步复制 Var 的 debug key；函数、region 的 Lam debug key
和正常重建的字段投影继续保留。
`grnames` 覆盖 flat x、grouped x/y、nested o 和 body 名字。
打印器仍可给重复名字加正常后缀，这不是 gid 生成名或丢失 source sym。

三个修复的完整验证分别为：C1 CTest 231/231、lit 216 通过；C2 CTest 231/231、
lit 217 通过；C3 CTest 231/231、lit 218 通过。
lit 均为 rules|error|tensor，均有原来的 4 个 unsupported，无失败或 XFAIL。

## Task 3 / 4：独立 frontend 回归与扩展

### Task 4 前置修复：深 DAG 与资源预算

graph rebuild、非重写 importer 与批量 FunctionPublisher 对 immutable DAG 使用显式栈，
继续复用 Rewriter 的 mutable stub / Var 映射；普通 op chain 不再受 256 层 rebuild 限制。
默认 work budget 按源 World 节点规模扩展，显式小预算仍可用于调用方资源限制。
耗尽返回可识别的 RebuildLimit，phase 丢弃 candidate / publication 并写 debug 诊断，
不把资源不足当成编译错误。recipe/dependency 的剩余深度预算同样走此拒绝路径。
同时替换仍按新的 recipe 依赖排序，不预遍历被替换 key 的旧 operands。

CTest 在真实 GraphRuleRewrite phase 上验证 400 / 5000 层 chain 全部保留且 q→v，
并对两种深度检查小预算时源函数不变、phase 不请求迭代。
lit 另验证 400 层源码的编译和输出，以及 5000 层源码的编译和 profile rewrite counter。
5000 层的 IR 结果用 C++ 逐节点检查，不依赖尚有递归展开限制的 Mim textual dumper。
验证：CTest 230/230；rules|error|tensor 211 通过、4 unsupported。
closed mutable type 的 native alpha Where 可能记录 snapshot-wide reads，仍会限制批量计划
之间的兼容性；本修复没有移除保守 read tracking。

### Task 4：region occurrence search / publication

graph frontend 在闭合 external function 中按 operand occurrence path 搜索 nested Lam bodies。
目前 owner 路径限 App、Tuple、静态 immutable Pack 和 Lam body，不搜索 filter/type，
也不跨 external function call 进入 callee。
region inputs 包括 enclosing binders 和该 body 的参数；GraphView 的 coverage 始终 Open，
显式 schema 可证明 known used_by edge，但 free variables 不是排他用途覆盖证明。

publication 用 fresh binder stubs 重建选中的 enclosing path，按冻结输入映射替换 leaf body；
未选中的 operands 用非重写 importer 保留原实现。
因此在 graph-fact/witness 的 occurrence 模式中，shared Lam 的 left/right 引用只改选中
的 body，其他 occurrence 保留原实现；纯规则的 value-wide 发布见前面的共享 region 修复。
`grregion` 同时覆盖两层捕获 offset、共享 closure 的 occurrence 隔离，以及 Open region
的 only_used_by 为 Unknown。immutable path 与 source DAG rebuild 都用显式栈。
每函数每次至多一个 changed rewrite、phase-wide source snapshot 与下一轮重新发现保持不变。
discovery 有随程序规模扩展的 work budget，耗尽时 debug 日志并跳过该函数。

`tf_local_g`、`tf_share_g` 的 NEST 和 `tf_rmap_g` 的 NEST 已翻转为目标检查。
这不是任意 mutable region、动态 Pack instance 或 alias/box witness 的通用机制。

### Task 4：binder-building RHS

graph Rule 的 Lam RHS 子树现在降为 instantiation recipe：在 prospective World 中，
把匹配得到的源 operands 冻结导入，再用非重写 Rewriter 的 stub / binder maps 构建新
Lam 及其 body。此 importer 不再次运行规则，不把新模板当成 source graph 的候选节点。
同一 recipe 保留同一构造结果；offset/accumulator 参数和 enclosing inputs 的映射被保留。
仍保留最终 source/replacement type check；C++ 测试同时验证 fresh Var、冻结 capture、
源 gid 不变和错误 result type 的拒绝。

所有 RHS 引入的 axiom calls，包括 nested body/filter 中的调用，都必须在 pure_ops 中；
实例化后再检查具体 argument groups 的 purity proof。
每层 curry prefix 也要保持为原 axiom 的应用，不能通过早期 normalization 偷换成另一个
function 后继续调用。失败返回 Unknown 并记录 debug 诊断，不发布 prospective World。
没有增加 arbitrary captured-function / Lam callee 调用、Hole 或依赖类型反向推断。
未证明纯性的计算 capture 不能移入新的 binder；已有 SSA 参数可作为 region input，
但把 effect(x) 这类计算藏进新 body 会被保守拒绝。
`grrhs` 验证缺少 inner-op purity 后仍可选择下一条合法规则，并保留 effect capture 的源程序；
`grdrop` 继续验证完全删除未证明纯性的 source computation 被拒绝。

`tf_build_g` 和 `tf_rmm_g` 已翻转为目标检查，生成三层 modeled/real contraction nest。
`tf_build_g` 也观察到 `stage _5` 打印在 io body 内的 mm 参数中：和 rule_app 一样，
这不是 placement 指令或 packing once-per-ko 的证明。
真实 `tf_rmm_g` 此输入把 bp 打印在 ko body，仍不能仅凭打印位置推断实际分配次数。

Task 4 最终验证：CTest **231/231**；rules|error|tensor **214 通过、4 unsupported**，
无失败或 XFAIL。B1 commit 为 230/230 与 211 通过；region commit 为 230/230 与 212 通过；
每个 commit 前都运行完整的上述两套测试。

以下更新优先于后面的 Task 2 历史结果。
每个 `tf_*.mim` 现在只运行 rule frontend，对应 `tf_*_g.mim` 只运行 graph frontend。
可工作的路径使用目标 CHECK；尚未工作的路径使用 unchanged IR 和精确诊断的正向 GAP 检查，
不再使用 XFAIL，避免 graph 的已知缺口掩盖 rule_app 回归。
`tf_share_g` 的 flat 多根和 nested NEST 均使用目标检查，rule 文件也独立保护 NEST。

`pure_ops` 新接受 qualified annex-name strings，与原有低阶 Axm 值可以混合。
例如 `("tensor.slice", "tensor.unary")` 在构造 phase 时解析为源 World 中实际的 Axm，
再与已有机制一样导入模板 World，重新运行时导入当前 World。
这选择了 annex references 路线，没有引入不存在的 Level 参数多态或改变类型检查器。
没有找到 annex、或引用非 Axm 时显式报错；没有自动推断 purity 或把未知 op 当作纯。
`tf_poly` 已翻转为正常通过；真实 graph probes 的配置均通过，剩下 LHS/guard/recipe 缺口
由各自 GAP 检查独立记录。`tf_run.py` 将真实 qualified op 转为这种显式引用。

实验 4 的 observed IR：rule_app 输出把 `stage _5` 打印在 io body 的 mm 参数中，
尽管 stage 只依赖 jo/ko。
这不是 rule_app 的错误；lexical let 不是 placement 指令，模型尚无 stage lowering 合同。

### Task 3：真实 LHS 的最小修复

临时在 MatchEngine 的 Exact 失败处记录 expected/actual 后，四个真实 probe 的首个阻挡
均是模板导入克隆后的 mutable Pi type 与源算子 type 的地址不同。
slice 在 `{T,r}` 部分应用之后的 dependent Pi、unary 在 f 部分应用之后的 implicit shape Pi、
product_2d 在 Ring 部分应用之后的 shape Pi 都出现此现象；不是 Pack metadata 或参数类型推断。
把 axiom 按 flags 匹配的试验只把失败推迟到下一层 partial-App Pi，不能单独解决问题。
最终没有保留临时日志或这个 axiom-only shortcut。

Rule adapter 现在仅对**闭合 mutable type**增加只读 `Check::test_alpha` 备选，
保留 exact 快路径、source World 跟踪、匹配预算、metavariable 的 deferred type 检查和最终
replacement type check。开放 dependent type、Hole 推断和 term-level lambda alpha matching
没有增加。这个修改不需要复制 tensor schema adapter，因为失败是共用 Rule 的 type identity。

`tf_rmap` 的 flat、共享 whole reader 和 tiled body 内 pushdown 都已通过 rule_app；
相应 `_g` 的 flat 已通过，但 nested 仍以 unchanged IR 检查 region 缺口。
`tf_rrev` 两个 frontend 都已生成 together；`tf_rmm` 的 rule_app 已生成真实 product/slice nest，
graph 仍正向检查 `unsupported graph Rule recipe value`。
第二个扩展使 `tf_rfact_g` 到达 guard；第三个扩展使它通过下述目标检查。
新增 length-16 negative case 防止 length-8 rule 误匹配；C++ 测试还检查导入 dependent Pi 的
匹配不会构造源 IR，并拒绝不同 axiom。

### Task 3：源 snapshot 上的 graph-fact guards

`compile.used_by` / `compile.only_used_by` 是 staged predicates，没有 ordinary normalizer。
只有 Rule guard evaluator 在有 `GuardContext` 时解释它们；普通函数内的调用仍是未求值的
Mim term（`grfacts`）。
数据 schema 与 purity 是两份独立的声明：

```mim
let schema = compile.data_op @((«12; I8»), 1) ("tensor.unary", 4, (3, 0));
extern lam _compile (): compile.Phase =
    compile.rule_app_with (exclusive, schema);
```

`data_op` 的三个字段是 op、完整物理 App group 数、`(group, tuple_depth)` 数组。
group 从 0 开始，depth=0 把该 group 作为一个数据输入，depth=1 解开一层 tuple。
unary 有四个 group，只将 group 3 的 tensor 输入作为 data edge，不能把 f/type/shape 当成
tensor reader。配置的 op 可为低阶 Axm 值，或 qualified annex-name string。
文件内的裸 `axm consumer` 并不注册为可查询的 annex，故模型使用实际的 consumer 值，
而不是猜测打印出来的 `probe.consumer` 名称。
静态 group 数、输入索引/深度、重复声明和遍历预算都被检查；配置错误显式报错。

另一入口是 `compile.graph_rule_app_with (rules, pure_ops, schemas)`。
schema 不授予纯性；pure_ops 不授予完整 data-use 信息。
没有 schema 的算子、未覆盖或 opaque 的用途、无法对应到源节点的 predicate operands 都是
Unknown，不触发规则。已知另一个 reader 或直接返回 producer 使 exclusivity 为 False；
`used_by` 则可在存在其他 reader 时为 True。guard 的其余纯计算仍在隔离 World 求值。

两个 phase 在一次 run 开始时建立一份 old-world GraphView。
Rule metavariable bindings 加上 LHS/实际 matched roots 的结构对应关系，冻结 predicate 的
源 operands；不通过在 source World 中重新构造 `producer x` 查找 hash-consed 值。
如果结构对应有冲突、缺节点或超过预算，guard 为 Unknown。
`grfrozen` 先把 producer 重写掉，再证明 parent guard 仍查询源 producer/consumer，
不是 bottom-up 重建后且可能已归一化的 target。

graph frontend 保留每函数每次至多一个 accepted changed rewrite，但先为所有函数选择
proposal，再一次性发布到新 World。
合并按 external symbol 重接 sibling/recursive functions，并保留其 binder；`grpub` 用
`@ff` 排除 beta unfolding 后检查这些引用。
`tf_guard` 的 first 删除共享常量 producer 的一个 reader，second 仍按同一次 run 的源
snapshot 保留原 consumer；不因先发布 first 就错误证明 exclusivity。
代价是保留各函数 prospective World 并多做一次整模块 publication。

这不是完整 region 实例机制：GraphView 在 Lam/Pi 边界停止，隐藏在 nested region 内的
operands/use 不能被宣称已完整覆盖。也没有新增原生 cover/preimage witness、动态
data schema 或任意 RHS 表达式到源节点的反向推断。

### 当前结果（Task 6）

所有文件都是正常测试，GAP 是正向检查已知限制，不是 XFAIL。
每个 frontend 独立验证，实际改写成功与“准确记录未改写”区分如下。

| 实验 | rule_app | graph_rule_app |
| --- | --- | --- |
| `tf_local`，modeled body 内 compute_at | 目标通过 | 目标通过（Task 4） |
| `tf_share`，保留 whole reader / tuple 多输出 | 目标通过，含 NEST | flat/VALUE/JOIN/NEST 通过（Task 4） |
| `tf_rev`，modeled reverse placement | 目标通过 | 目标通过 |
| `tf_build`，整棵 cache_read nest RHS | 目标通过 | 目标通过（Task 4） |
| `tf_post`，后置 stage 插入 | 目标通过；stage 在 inner mm 参数中打印 | 同左（Task 6） |
| `tf_pack`，persistent marker | SSA 替换/共享通过 | SSA 替换/共享通过 |
| `tf_guard`，source graph facts | single/used 目标通过，shared/unknown 保留 | 同左，且所有函数同一 snapshot |
| `tf_poly`，多态 purity 引用配置 | 不适用 | 配置与目标通过 |
| `tf_rmap`，真实 slice/unary | flat/shared/NEST 目标通过；extent mismatch 不重写 | flat/shared/NEST 通过，mismatch 不重写（Task 4） |
| `tf_rrev`，真实 unary producer/reader | 目标通过 | 目标通过 |
| `tf_rmm`，真实 product/slice nest RHS | 目标通过 | 目标通过（Task 4） |
| `tf_rpost`，真实 slice 的后置 stage | 目标通过；outer jo 的 bp 被 inner read 使用 | 同左（Task 6） |
| `tf_rpack`，真实 slice 读取 packed 值 | SSA 替换/共享通过 | SSA 替换/共享通过 |
| `tf_rfact`，真实 unary single-consumer guard | single 目标通过，shared 保留 | 同左 |
| `tf_wbox`，mutable map 的 producer box | 目标通过，unsupported map 保留 | 同左（Task 6） |
| `tf_rbox`，真实 transpose 的 producer box | guard-proven shape specialization 通过 | 同左（Task 6） |

R2 的 placement 观察仍然成立：`tf_build` 里的 `stage _5` 在 io body 打印出来，
不能由任何这些 CHECK 推出每 jo/ko 只执行一次、persistent 缓存或 GEMM 性能。

### 剩余扩展的顺序

| 顺序 | 最小扩展 | 解锁范围 |
| --- | --- | --- |
| 1 | Task 7 的 map_reduce_post outs/map_init、tiled/tile_fold/stage 参考语义与 lowering | 所有实验的数值、placement、exactly-once 与 persistence 合同 |
| 2 | 通用 cover/preimage/affine/broadcast/layout callbacks、更多 data schemas 与重算预算证明 | 真实 compute_at/reverse placement、shared producer 和 cache_read 合法性；本轮只提供 registry 与 permutation box |
| 3 | staged-dependent shape refinement/typed recipes 与更一般的 binder 对应关系 | 非 specialization 的动态 producer box，以及超出同数/同深度 nest 的规则 |

data-use schema 当前必须显式提供，native cover/preimage callbacks 仍需按具体调度补齐。
本轮没有实现 Task 7；P5 的这些扩展仍不是 C++ tile_fuse 的数值/性能替代品。

第三个扩展最终验证：CTest **229/229**；`rules|error|tensor` **210 通过、4 unsupported**，
无 XFAIL、无失败。unsupported 仍是两个 error fixture、Linux-only `free_temps` 和
Release-only `vgg16`。

## Task 2 历史记录（以下不是当前结果）

### 范围与复现

基线为 `cd8526dd22` 加 Task 1 的两个修复和 `031b1d7626`。
本轮只修复拒绝绑定后的回溯；没有增加 region、lambda recipe、图 guard 或调度 lowering。
实验在 Debug 构建运行，使用 `compile.rule_app` 和 `compile.graph_rule_app`。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build \
    --target mim mim_all_plugins mim-test mim_lit_tests mim-regex-test mim-tensor-pattern-test
ctest --test-dir build --output-on-failure
cd lit
/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
    ./lit ../build/lit -a --filter 'rules/tf_'
```

`lit/rules/tf_run.py` 为同一个输入分别追加两个 frontend 的 `_compile`，运行一次
`phases ff` 和 cleanup，并将 IR 与 debug 日志分开。
生成文件使用短 module 名，退出时清理；不是新的编译器机制。
`TF-RULES` 和 `TF-PURE` 是该测试驱动的配置，不是 Mim 语法。
可单独复现某个 frontend，例如：

```sh
python3 lit/rules/tf_run.py build/bin/mim lit/rules/tf_build.mim build/tf_probe rule
python3 lit/rules/tf_run.py build/bin/mim lit/rules/tf_build.mim build/tf_probe graph
```

每个有缺口的测试保留正向的目标 IR 检查并标记 `XFAIL: *`。
因此“XFAIL”不是语法报错就算完成：模型的 single-root 成功检查在 graph 缺口前运行；
真实多态算子的已知类型错误另以 `not` 和诊断文本检查，再检查尚未得到的目标 IR。
XFAIL 文件不会继续执行第一个失败的 RUN，下面的分项结论也依据独立的 frontend 输出。

这些是**表达能力与 IR 结构实验，不是数值、bufferization 或性能实验**。
模型的 `Nat` 是不透明 tensor handle，不是 tensor 元素值。
`slice`、`panel`、`tiled`、`fold`、`stage` 等的含义由文件中的短注释约定；
没有 normalizer/lowering 来执行这些约定。
尤其 `tiled_n` 返回两个值不等于已经实现同一个 buffer；
`stage_persistent` 被列为 pure，只代表一个可引用的资源值，不证明分配、初始化或跨调用缓存纯。

最终验证：CTest **228/228**；`rules/tf_` **4 通过、10 XFAIL**；
全部 `rules|error|tensor` **181 通过、10 XFAIL、4 unsupported**，无意外失败或 XPASS。
unsupported 是两个 error fixture、Linux-only `free_temps` 和 Release-only `vgg16`。

## 逐项结果

### 1. Local compute_at

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `slice(ew(f,x),o,e) → ew(f,slice(x,o,e))`，位于捕获 `x/f` 的 tiled body | rule_app | **通过**；`tf_local.mim` 的 LOCAL 检查看到 body 内先 `slice(x,o,2)`，再 `ew(f,…)` |
| 同一规则、同一输入 | graph_rule_app | **XFAIL**；编译成功但 `Complete (no accepted match visited)`，body 中仍是 `slice(ew(…),o,2)` |
| 实际 `tensor.slice` / `tensor.unary`，含 offset 参数 | rule_app | **XFAIL**；`tf_rmap.mim` 中仍先 unary 再 slice，没有目标 pushdown，也没有 Unknown-plan 日志 |
| 同一真实算子 | graph_rule_app | **XFAIL**；配置在进入 phase 前报 `argument is not assignable to callee's domain` |

第一项推翻了“两个 frontend 都不能进入开放 Lam body”的预期。
`RuleRewrite::rewrite_imm` 使用 enclosing Rewriter，绑定已有变量后替换
（`src/mim/phase/rule_rewrite.cpp:60–82`；
`src/mim/pattern/rule.cpp:134–162`）。
GraphView 却在 Lam/Pi/mutable 处停止原始 operand 遍历
（`src/mim/pattern/graph.cpp:46–55,65–79`），phase 只选择闭合 external Lam
（`src/mim/phase/graph_rule_rewrite.cpp:65–96`）。
这里只缺 graph frontend 的 region 查询与发布，不应重写已经可用的 single-root 路径。

### 2. Producer 同时被 whole reader 使用

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| local 规则只重算 slice，保留 `whole(ew(f,x))` | rule_app / graph_rule_app | **通过**；`tf_share.mim` 的 VALUE 检查分别保存 tile 与 whole producer |
| 同样的 value-correct 形式，tile 在嵌套 Lam 内 | rule_app | **通过**；NEST 检查看到 tiled body 的 pushdown，whole reader 仍存在 |
| 同一嵌套形式 | graph_rule_app | **未重写**；与实验 1 相同的 region 缺口 |
| `(tiles(p),whole(p)) → tiled_n(p)`，结果恰好是共同 tuple | rule_app / graph_rule_app | **通过** |
| 两个 anchor 位于不同深度的 sink 参数，没有共同的 LHS tuple | graph_rule_app | **通过**；JOIN 检查 `multi_nested` 中出现 `tiled_n`，原来的两个 reader 消失 |
| 上述无共同 tuple 形式 | rule_app | **未重写**；single-root 入口不会把两个独立节点组成匹配根 |
| 实际 unary producer 的 slice 与 whole 输出 | 两者 | `tf_rmap.mim` 的 `shared` 仍受实验 1 的真实算子匹配 / pure_ops 类型缺口阻挡 |

图规则能表达“共享一个新逻辑值”，但尚不能表达 cover witness、realization cache 或
`tiled_n` 的目的 buffer、初始化顺序与寿命。
VALUE 的 whole producer 被保留，不代表 tiles 与 whole 不重复计算。
本实验也没有声称任意 map、非纯 producer 或 reduction 都能使用这个等式。

### 3. reverse_compute_at

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `(tiled(e,body),consume(tiled(e,body))) → tiled_n(e,body)` | rule_app | **通过**；本输入的 sink 参数本来包含共同 tuple |
| 同一规则 | graph_rule_app | **通过**；`tf_rev.mim` 的 REV 检查看到新 pair，consume 消失 |
| 实际两个 `tensor.unary` 的 producer/reader，RHS 为 modeled `together` | rule_app | **XFAIL**；`tf_rrev.mim` 保留两个 unary |
| 同一真实算子 | graph_rule_app | **XFAIL**；polymorphic pure_ops 类型错误 |

`body` 是从源图捕获的函数值，不是新建 lambda；这解释了 graph recipe 为什么能接受它。
共享 capture 可以编码“C 的 operand 是 P”这一特例，不需要 guard。
它不能代替一般的 `used_by`、精确 preimage、其他输入不依赖 P、完整覆盖以及 reduction-order witness。
若 reader 跨 tile 维归约，需有 accumulator continuation 的 `tile_fold`，本实验没有模拟其合法性。

### 4. 构造时 cache_read

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `mark(mm(a,b))` 生成 jo tiled → ko fold → `bp=stage(panel(b,ko,jo,2))` → io tiled | rule_app | **通过**；`tf_build.mim` 的 BUILD 检查覆盖 panel、stage、两个 tiled 和 fold，mark 消失 |
| 同一规则，三层新 lambda | graph_rule_app | **XFAIL**；`unsupported graph Rule recipe value`，原 mark/mm 保留 |
| 实际 `product_2d`、矩阵 slice 与 modeled region/stage 的同形 RHS | rule_app | **XFAIL**；`tf_rmm.mim` 仍为 mark(product_2d)，没有 stage |
| 同一真实 RHS | graph_rule_app | **XFAIL**；pure_ops 类型错误先于 lambda recipe 验证 |

所以“single-root 不能生成带 binder 的 RHS”也不成立。
但 lexical `let bp` 并不是 placement 指令：IR printer 可以在 inner body 中内联 `stage`，
而资源应在哪个 region 实例执行仍需 lowering 保证。
模型 fold 的 0 约定为零 accumulator handle；真实实验使用正确的 2×2 A/B panel，
把 accumulator 的行 tile 加到新的 product 上，但该 fold/rows 仍无执行实现。
不能由 BUILD 通过推出 packing once per ko tile 或 equal GEMM performance。

### 5. Post-hoc cache_read

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| 匹配已有 jo/io nest，把 B tile 的 stage 放到 jo body | rule_app | **XFAIL**；`tf_post.mim` 没有 stage |
| 同一 lambda-rich LHS | graph_rule_app | **XFAIL**；独立运行得到 `Complete (no accepted match visited)` |
| 内部读取改为实际 `tensor.slice` 的同一 nest | 两者 | **XFAIL**；`tf_rpost.mim` 没有 stage，graph 配置自身可通过类型检查 |

这里不是仅仅增加 RHS lambda 就能解决。
`RulePattern::lower` 将 mutable 节点及非 Rule Var 变成 exact pattern
（`src/mim/pattern/rule.cpp:47–52`），新的 LHS lambda 不会与另一个实例做 binder 对应匹配。
需要 alpha-aware LHS binder 捕获，以及跨两个 region 的 scope/environment 映射和事务。
若提前留有外层 stage marker，可做局部单根替换；但这不等价一般的后置 box union 与 inner map rewrite。

### 6. Persistent prepack

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| 函数级 `frozen(b) → stage_persistent(b)`，两个 slice 读取此值 | rule_app / graph_rule_app | **通过**；`tf_pack.mim` |
| 同一 marker 规则，slice 换成实际 `tensor.slice` | rule_app / graph_rule_app | **通过**；`tf_rpack.mim` 的 PACK 检查绑定一个 packed 值，两次实际 slice 都读取它 |

真实 slice 不被改写，也没有被声明 pure，因此无需把其多态 axiom 放进 pure_ops。
这不是绕过 purity：slice computation 保留在重建后的图中。
实验只证明 frozen marker 能被替换且两个读取共享同一个 SSA 值。
冻结条件、权重版本、设备/布局键、缓存初始化/释放、并发调用以及跨 invocation 复用仍完全缺失。
该分支的 tensor API 没有设计文档中的 `stage_persistent` 资源 op；本文件只建模它。

### 7. Graph-fact guard

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `consumer(producer(x))` 的 guard 调用 modeled `only_used_by` | rule_app | **XFAIL**；`guard did not normalize to true` |
| 同一规则，分别输入 single / shared reader | graph_rule_app | **XFAIL**；两者都是 `guard is not true`，没有虚构 single-consumer 成功 |
| 实际 unary 输入的 `only_used_by` guard | rule_app | `tf_rfact.mim` 中仍为 unary，尚未到可用的 guard 实例 |
| 同一真实图 | graph_rule_app | **XFAIL**；pure_ops 类型错误先于图 guard |

guard importer 只在隔离 World 中做 term 求值
（`src/mim/pattern/graph_rule.cpp:75–91`）；这里没有 GraphView 查询入口。
C++ `GraphView::data_only_used_by` 和 GraphPattern 的关系接口已经存在；
但 phase 对 `pure_ops` 的注册没有数据 schema，并将完整性设为 false
（`src/mim/phase/graph_rule_rewrite.cpp:70–91`）。
不能把“有纯性声明”当作闭合、完整的 tensor data-use 信息。
给 only_used_by 写一个普通 normalizer 也不能正确查询随事务变化的 snapshot。

## 真实算子还揭示的两个前置缺口

1. **polymorphic pure_ops 的入口类型。**
   `compile.graph_rule_app` 的 `O` 当前是 `*`，而实际 `tensor.slice` /
   `tensor.unary` 含类型参数的函数类型需要更高 universe
   （`src/mim/plug/compile/compile.mim:136`；
   `src/mim/plug/tensor/tensor.mim:312–316,383–387`）。
   `tf_poly.mim` 把规则换成 Nat identity，只传 singleton `tensor.slice`，仍得到
   `expected [Rule Nat, ?], got [Rule Nat, {T: *, r: Nat} …]`。
   因而不只是 heterogeneous tuple 的问题，也不是 tensor matcher、RHS 或 guard 引起的错误。
   尝试 `refly.type` 推出显式 O 也在该 helper 的 `T: *` 入口遇到同类错误。
   最小修正方向是使配置对 rule/pure-op 值的 universe 多态，或用 staged annex 描述符列表；
   不是将 tensor 算子谎称为低阶函数，也不是补齐 purity 后就停止验证。

2. **真实多态 App 的 Rule LHS 匹配。**
   rmap/rmm/rrev 的 rule_app 编译成功，但没有目标重写。
   当前 Rule adapter 对每个结构节点也匹配 type，mutable/ground 节点使用 exact identity
   （`src/mim/pattern/rule.cpp:37–59`），这不是 tensor op schema 加 alpha-equivalent
   dependent-type matching。
   本轮仅确定“LHS 没有产生可替换匹配”，未给每个失败的中间 App/type 子模式加 instrumentation；
   不把“所有 array capture 均不支持”当作结论，rpack 的成功已否定这个说法。
   下一小步应隔离部分应用的 Pi、隐式类型组和 Pack metadata，复用
   `src/mim/plug/tensor/pattern.cpp` 的真实 op adapter，保留最终替换类型检查。

规则 telescope 或函数参数的 tuple η 归约是另一个实验干扰因素：
仅有 `(a,b)` 时 `mm(a,b)` 可能变为 whole-telescope capture，RHS 单独引用 a/b 会缺绑定。
模型构造规则显式多一个未使用字段，候选函数也保留 salt，以隔离 lambda 能力；
不声称这修复了 whole-telescope/projection 支持。
现有 `validate_bindings` 不能反向构造缺少的 capture
（`src/mim/pattern/rule.cpp:79–101`）。

## 最小扩展的优先级

以下“解锁”指移除一个必要阻挡，不承诺单项扩展就完成调度与 lowering。

| 优先级 | 最小扩展 | 涉及实验 | 边界 |
| --- | --- | --- | --- |
| 1 | Universe-safe 的 graph phase 配置 | 真实 1/2、3、4、7，4 个 probe 文件；`tf_poly` 隔离入口 | 只移除配置类型错误；6 已可通过，不应强迫把未改写的 slice 列入 pure_ops |
| 2 | 真实 tensor Rule LHS 的 schema/type 匹配适配 | 同上 4 个 probe 的 rule_app 未匹配问题 | 先定位中间 type/metadata 子模式；不取消最终替换类型检查 |
| 3 | Region-local GraphView + 带捕获环境的 scope publisher | 嵌套 1/2，以及后置 5 的 region 部分 | owner/environment 必须保留 offset、accumulator 和 type 依赖，不能仅把 Lam body 加入原始遍历 |
| 4 | 带新 binder/环境的 RHS recipe，仍在独立 World 构造与验证 | graph 4，以及 graph 5 的构建部分；真实 GEMM 的后续生成 | single-root 已能建 lambda；不要把它当作两个 frontend 共同缺口 |
| 5 | Alpha-aware 的 LHS binder matcher，加 region 间 box/map 事务 | 模型/真实 5 | 只给 lambda wildcard 会丢失 offsets 与内部 use 的约束 |
| 6 | Staged snapshot guards / native witnesses，真实 tensor data-use schema | 模型/真实 7；2/3 一般合法性 | Unknown、coverage 和 witness generation 要显式；不放入全局纯 normalizer |
| 7 | tiled/tile_fold/stage/outs 的实际 lowering、realization 与持久资源协议 | 所有实验的数值、共享和性能含义，尤其 2/3/4/6 | 这是 tensor/storage 语义，不是 P5 matcher 的小补丁，另列为性能实验的前提 |

**结论：** 用现有 single-root 路径可先做“整棵模型 nest 的规则生成”；
用现有 graph 路径可做“闭合函数内、lambda-free recipe 的多根值替换”。
不能以这两项通过宣称 C++ tile_fuse 已被替代：
真实算子适配、跨 region 合法性，以及 stage/accumulator 的执行合同仍需分别验证。
