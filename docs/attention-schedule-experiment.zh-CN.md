# Attention 调度机制实验：先预测，再测量

## 实验边界

计划基线为 `1fa84179bc7ad6ad24d0928f93eafbefe0342fd4`。
本节在执行计数、基准或调度实验之前写下；后续结果单独追加，不覆盖原预测。
先重建该基线，研究目标是验证 P5 规则、tensor tile 构造和 group 选择的正确性与可行性，
不是通过调整默认参数追求某个延迟。
计时均为单线程，预热后至少三次取中位数；并发编译下不解释小幅差异。

实验输入为 gemm-bench 的 attention 与 transformer 程序，
batch=1、heads=12、embed=768、head_dim=64、hidden=3072，
L=128/512/1024/2048。另用小型非整块尺寸测试 causal/non-causal、尾块和数值稳定性。
所有比较使用相同输入；eager 不必采用相同的内部实现。

## IREE 源码依据

- `Dialect/LinalgExt/Transforms/TileAttention.cpp` 实现
  `ConvertAttentionToOnlineAttentionPass`：从 attention 创建 output/max/sum 三个
  FP32 carry，转换为 online attention，在外面做最终归一化。
- `Dialect/LinalgExt/Transforms/DecomposeAttention.cpp` 在已经切好的
  `OnlineAttentionOp` 内调用 `decomposeOperation`，不是先生成整个 scores 再切片。
- `Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp` 实现 carry 更新：
  `newMax=max(oldMax,rowMax(S))`、`norm=exp(oldMax-newMax)`、
  `P=exp(S-newMax)`、`newSum=norm*oldSum+rowSum(P)`、
  `newAcc=norm*oldAcc+P*V`。其可选 exp2 路径还会修正 scale。
- `Preprocessing/Common/FoldAttentionWithTranspose.cpp` 把输出的
  expand/transpose 合并进 attention 的 indexing maps；要求单一使用并验证映射。
  这说明布局应作为读取映射处理，不应靠识别 Q/K/V 的变量名。
- `Codegen/LLVMCPU/KernelDispatch.cpp` 的
  `setRootConfig(OnlineAttentionOp)`：batch 分布 tile=1；M/N 分布；
  静态 K1<=128 不切；K2 按 vector size 切；再分别用 QK/PV 约束寄存器空间。
- `Codegen/LLVMCPU/Passes.cpp` 的
  `addCPULinalgExtTileAndVectorizePipeline` 先 tile/distribute、tile/fuse，
  再 decompose attention，然后 vectorize、bufferize。

以下采用相同的“保留复合语义 → 切 tile → tile 内分解”顺序。
不假定 MimIR 与 IREE 的 leaf 或运行时等价。

## 计数口径与基线预测

只对 attention core 计数，另列完整模型，不能把完整模型延迟分摊成某个 GEMM 的 GFLOPS。
记 h=12、D=64、N=D、score 元素数 E=h*L*L。
QK 和 PV 各约 `2*h*L*L*D` FLOP，总计 `4*h*L*L*D`；
精确乘加次数分别报告，FMA 按两 FLOP。
普通 softmax 的 exp 次数应为 E，非 GEMM 的算术量为 O(E)。
scale、减 max、sum、div 的融合可能减少写流量，但不应重复整个 QK。

| L | QK+PV GFLOP | exp 次数 | 每个完整 S/P 的字节数 | 同时保存 S、P 的字节数 |
|---|---:|---:|---:|---:|
| 128 | 0.050331648 | 196608 | 786432 | 1572864 |
| 512 | 0.805306368 | 3145728 | 12582912 | 25165824 |
| 1024 | 3.221225472 | 12582912 | 50331648 | 100663296 |
| 2048 | 12.884901888 | 50331648 | 201326592 | 402653184 |

上表最后一列不是对现有编译器峰值的断言：
实际可能因释放时机、softmax 复用或额外中间数组而不同，必须记录分配与释放验证。
Q/K/V 和输出各为 `4*h*L*D` 字节，属于线性项；
packed QKV 与 projections/MLP 的存储单独归账。
除峰值外记录累计申请字节数、申请次数和每类缓冲区的生命周期。

## 步骤 1：raising 规则

预测结构：P5 graph pattern 匹配 QKᵀ→scale→可选 mask→稳定 softmax→PV 的共享 DAG，
变成一个 attention construct，携带逻辑形状、读取映射、scale 和 mask 语义。
采用 tensor-level construct，使 tile 前保留稳定 softmax 的复合语义，
避免现有 GEMM/elementwise 融合提前破坏 diamond；
construct 的普通分解必须能恢复原表达式。
规则不得依据变量名、模型名或某个 L 匹配。

预测计数：只做 raising 并普通分解时，GEMM FLOP、exp 次数和中间存储阶数不变；
可能出现等价融合造成的分配差异，但不能消除二次方 score 存储并声称已实现 online。
性能预测：普通分解相对原程序为 0.8–1.25 倍；
超出范围首先检查布局和 leaf 变化，不归功于复合 op 本身。
验收：attention、transformer 各恰好 raising 一处；错误 scale 位置、
不稳定 softmax、错误 reduction axis、非乘加 contraction、不同 exp 输入、
额外可观察 reader 等 near miss 不得被不合法地吸收。

## 步骤 2：online 规则

预测结构：运行时 batch grid（tile=1）→M tile→K2 的顺序 `tile_fold`；
carry 为 `(row_max,row_sum,output_accumulator)`；
K1=64 不切，最后一个 K2 尾块只读有效元素。
每轮只生成当前 `BM*BK` scores/probabilities；
QK/PV 在 tile 内分解，能力允许时使用现有 BLIS/register-blocked leaf。
fold 之后做一次 output/sum，绝不在每个 K2 block 上归一化输出。
不得把 batch 各点静态复制成独立 nest。

首次实验固定 BK=目标 vector lanes（当前预测 8），BM 从候选 1/2/4 中选满足
`BM*(D+BK+2) <= registers*lanes` 的最大值，预测 BM=2。
这是实验参数，不是全局默认调优；文档必须保存实际选择及其理由。
N=64 的 accumulator 在 tile 内保留；若 bufferization/leaf 无法保留，记录偏离。
不把“使用 BLIS”当作正确性前提。

预测计数：主要 GEMM FLOP 不变；score exp 仍为 E；
额外 rescale exp 最多 `h*L*ceil(L/BK)` 次（首块可消去），
BK=8 时约增加 12.5% exp。
carry 缩放与合并约增加 `O(h*L*ceil(L/BK)*D)` 算术。
causal 首次实现允许计算全部 score 再 mask，因此不承诺 FLOP 减半；
所有行至少有对角线有效，另测全 mask 行的显式语义，不能让 `-inf-(-inf)` 偷渡。

理想峰值工作区（不含输入/输出和 BLIS 面板）不超过
`4*(2*BM*BK + 2*BM*D + 4*BM)` 字节，
BM=2、BK=8、D=64 时为 1184 字节；
额外一个 carry 副本或 Q/K/V panel 必须逐项计入，不能隐去。
容许累计申请字节数仍为 O(L²)，但活动峰值不能再含完整 h*L*L 数组。
输出累计写回量可能随 fold 次数增长，应与主要计算量区分。

性能预测而非目标：若 leaf 与原 BLIS 同等有效，按 40–100 GFLOPS，
纯 GEMM 下界在四个 L 约为 0.50–1.26、8.05–20.13、32.21–80.53、
128.85–322.12 ms，另加 exp、映射和分配成本。
BK=8 的小 contraction 可能明显降低 leaf 效率并引入 fold/分配开销，
因此初始 online core 延迟预测为当前 BLIS core 的 0.7–5 倍，
而非保证更快。大 L 的二次方全局中间存储消失是主要可证伪预测。
小 tile 慢于 unfused 但结构/计数符合预测，仍是机制可行性的结果。

## 步骤 3：手动和自动选择

预测结构：手写 schedule 显式应用 raising 和 online 规则；
group 阶段选择同一规则和同一参数，只改变选择入口，不维护第二套实现。
两者 realized dump 的 batch/M/K2 trip count、carry、leaf、stage 完全一致；
不比较 gid 或 LLVM 变量名。
相同输入数值在 FP32 容差内一致，标量参考包含 causal/non-causal、
大正负 score 和非整块尺寸；初始容差为 atol=2e-4、rtol=2e-4，
不因一次失败直接放宽。

预测计数：相同 FLOP/exp、分配次数、累计/峰值字节。
预测时间：同进程、交替顺序、各至少三次中位数相差不超过 10%；
若置信不足增加重复，不能把并发负载差异当作机制差异。

## 步骤 4：对照矩阵与归因

每个 L 比较 group=0、当前自动（仅禁用新增 attention 规则）、
新自动、手动、PyTorch eager。
attention/transformer 完整模型都测；同时测孤立 core，避免 projection/MLP 掩盖结果。
记录编译 SHA、命令、编译失败、准确性误差和实际 rule/leaf 计数。
计数版不用于延迟；非插桩版使用同一输入和线程设置。
allocation 插桩区分 malloc/alloca、分配容量与有效元素，exp 插桩处理向量 lanes，
不只数静态 IR 里的 call。
若优化器复制表达式，动态计数必须暴露它。

预期 L 翻倍时主要 FLOP/score-exp 约四倍；
online 全局活动存储的 L 依赖只来自输入/输出的线性项，
非在线对照可能保留二次方中间数组。
完整 transformer 的线性 MLP 项会使总延迟增幅小于四倍。
测量表逐步列“预测 / 测得 / confirmed 或 deviates / 原因”，保留反例。

## 步骤 5：表达能力报告

预计图识别、语义 guard、结构替换和参数化 tile/fold 可以用 P5 规则表达；
标量/布局证明可能需要 witness，不等于允许在 C++ 中隐藏整套调度。
需要实际核实异构三元 carry、运行时 tile 生命周期、register-space 限制、
bufferization、向量 legalization、exp2 选择和 CPU 分布中哪些不能由当前规则表达。
无法表达时给出最小缺口和失败诊断，不把 IREE 的全部底层 pipeline 冒充为一条规则。
每步独立提交，提交前运行 CTest 与完整 lit；遇阻按约定停止并报告。

## 重启后的实验状态：步骤 1 阻塞，尚未测量

计划提交为 `8618a6bd808e9d5c975881efca09624051efd653`。
该提交之前 CTest 243/243 通过，完整 lit 673 通过、9 unsupported。
机器重启后按四核上限重新构建成功；所有新日志和复现保存到
`/Users/zc/courses/compiler/products/tile-fuse/results/task28a/`，
不依赖临时目录或旧 scratchpad。
后续重任务限定 `nice -n 10`、最多四核且串行，计时仍为单线程。

当前未提交的实现尝试用一个 tensor attention axiom 保存复合语义，
用 P5 graph rule 匹配由 `attention_impl` 展开的原始 diamond，
普通 lowering 则重新分解它，尚未引入 online 算法。
但是 raising 尚未进入匹配或替换，规则自身编译就失败：

```text
Rule LHS bound function application requires higher-order matching
```

`raising.log` 保存完整诊断。LLDB 在 `src/mim/pattern/rule.cpp:184`
截获实际被拒绝的表达式为 `mk (3, 0)`，
见 `raising-expression.log`。
它来自展开后 softmax 的 `map_reduce_post` 默认调度：
`SchedT` 以 `λ T. λ mk. mk (3,0)` 编码。
`RulePattern::lower` 遍历 LHS 的 lambda body 时，
会拒绝所有 callee 为 lambda-bound Var 的 application，
即使外面的整个 schedule 是闭合、固定的值。
因此当前这种直接展开 diamond 的结构规则不能编译；
此处不是 shape mismatch、未命中模型或 online 数值误差。

独立最小复现 `schedpat.mim` 只包含一个接受闭合 schedule 的 `mark` axiom、
`λ T. λ mk. mk 3` 和删除 mark 的规则，不包含 attention 或 tensor。
运行以下命令同样以状态 1 报上述诊断：

```sh
cd /Users/zc/courses/compiler/mimir-gpm
nice -n 10 build/bin/mim \
  /Users/zc/courses/compiler/products/tile-fuse/results/task28a/schedpat.mim \
  -o /Users/zc/courses/compiler/products/tile-fuse/results/task28a/schedpat-output.mim
```

这是一项具体的表达方式限制，不证明所有 P5 raising 方案均不可能。
可研究的替代是把 schedule 作为有语义 guard 的 opaque capture，
或让 pattern compiler 安全比较闭合 schedule 而不递归解释其 binder 应用；
尚未实现或验证这些方案，不能将 witness 内隐藏整个识别器当成已经解决。
当前 WIP 还未覆盖 mask、真实 attention/transformer 命中或完整 near-miss 集合，
且已接入默认流水线，不能视为可用版本。
按任务的遇阻停止约定保留 WIP、不提交步骤 1。
步骤 2–5 和计数/时间测量均未进行，因此没有 predicted-vs-measured 的成功结论。

### 闭合 schedule 的修复与后续复现

上述编译阻塞已由 `4d8abc29db` 修复：
`RulePattern::lower` 在结构化 lambda 之前识别闭合 lambda，
以 exact 或 alpha-equivalence 比较常量；`lit/rules/closedlm.mim` 覆盖该情况。
这是机制层面的实质发现：调度虽然是函数编码，但闭合调度在 LHS 中应作为常量，
不需要求解 `mk` 的高阶应用。
本轮确认 attention 规则已不再报原来的高阶匹配错误。

新 op 的导出名称改为 `tensor.sdpa`，避免与已有 `use tensor` 程序中的
局部 `attention` 定义冲突。`tensor/attn_exec.mim` 已重新通过。
raising 仍未匹配；即使诊断时把全部形状和 scale 固定成字面量，
正确 diamond 的直接 `RulePattern::match_lhs` 也返回 No_Match。
诊断用固定尺寸、源代码路径和 dump 调用已经撤掉，WIP 恢复通用参数。

进一步缩小后发现闭合 lambda 常量的重复出现仍附带节点身份约束。
最小复现为结果目录的 `clrep.mim`：

```mim
lam expected (x: Nat): Nat = core.nat.add (x, 1);
lam first (x: Nat): Nat = core.nat.add (x, 1);
lam second (x: Nat): Nat = core.nat.add (x, 1);
rule remove (x: Nat): mark (expected, expected, x) => x;
```

使用 `compile.graph_rule_app`，并显式声明 mark 与 core.nat.add 的纯性后：

| 目标 | 预期（闭合常量按 alpha 比较） | 实际 |
|---|---|---|
| `mark(first, first, x)` | `x` | `x` |
| `mark(first, second, x)` | `x` | 未替换 |

`clrep-graph-output.mim` 和 `clrep-graph.log` 保存输出及匹配日志。
同样的现象也已在 `compile.rule_app` 中复现。
原因是闭合 lambda 分支的 alpha predicate 使用一个 capture；
`lowered_` 复用该 pattern 时也复用 capture，
而 `match.cpp` 的 Capture 分支要求后续绑定与已有绑定指针相同。
因此“一个固定函数常量出现两次”被额外解释为“目标两次必须共享同一个 mutable”。
这与 alpha 常量比较不同，会影响展开 diamond 中重复的默认 schedule、
identity post 和访问映射；尚不能宣称修复该点后整个 raising 一定通过。

执行最小复现：

```sh
cd /Users/zc/courses/compiler/mimir-gpm
nice -n 10 build/bin/mim \
  /Users/zc/courses/compiler/products/tile-fuse/results/task28a/clrep.mim \
  -o /Users/zc/courses/compiler/products/tile-fuse/results/task28a/clrep-graph-output.mim \
  -VVVV
```

暂停状态：已重建，定向 lit 为 attn_exec 通过、attention_raise 失败；
记录于 `paused-build.log`、`paused-targeted.log`。
没有新提交，没有模型测量，也没有 online 实现。
按任务要求遇到新的、独立可复现的机制阻塞后停止，保留步骤 1 WIP。

### 重复闭合常量修复与匹配预算阻塞

第二个阻塞由 `3f7ebc465b` 修复：闭合常量 pattern 不再被 memoize，
每次出现使用独立的 alpha predicate/capture。
本轮重跑 `clrep.mim`，`shared` 与 `distinct` 都返回原参数，
输出保存在 `clrep-fixed-output.mim`；`rules/closedlm.mim` 也通过。
因此前两项机制修复均已验证，不应再将后续未命中归因于它们。

用户指出 op 内嵌 schedule closure 是设计异味：
计算应与调度解耦，调度由规则施加。
此项重构已记录，但不在本任务启动；当前实验继续使用现有 op。

本轮还识别到 `map_reduce` wrapper 没有显式给出零个 post 输入的类型/秩/形状，
默认推导会留下互不相同的 Hole。直接结构匹配会把这些 Hole 按身份比较，
即使它们表示的是同一空 tuple。
WIP 已将 wrapper 的 `(nis,0)`、完整类型组和 `(Tis,Ris,Sis,(),(),())`
写明；未修改 P5 matcher，也没有给不确定 Hole 增加宽松匹配。
该变更尚未全量验收或提交。

排除上述问题后，诊断用的全字面量形状与 scale 规则实例，
对正确 attention 根直接调用 `RulePattern::match_lhs`：
结果为 `Budget_Exceeded`，steps=660，见 `explicit-metadata.log`。
默认 max_steps 为 100000、max_depth 为 256；
当前 `match.cpp` 同时将结构 depth 和 continuation 的 active_depth 计入该深度预算。
因此这不是总 steps 用尽，而是深度限制。
扁平 tuple 的兄弟分支也通过 continuation 嵌套执行，预算并不只衡量源表达式的树高。
未擅自提高默认上限，也未把诊断中的固定形状留在生产代码。

独立复现为结果目录的 `pdepth.mim`：
规则把 `mark(x,x+1,…,x+31)` 改写为 x，目标正是同一表达式，
没有 tensor、softmax、schedule 或闭合函数常量。
16 元素的同构对照能改写；32 元素版本报
`rule_app exceeded its pattern matching budget`。
干净 core（已撤掉所有临时 dump）仍报错，见 `pdepth-final.log`。

```sh
cd /Users/zc/courses/compiler/mimir-gpm
nice -n 10 build/bin/mim \
  /Users/zc/courses/compiler/products/tile-fuse/results/task28a/pdepth.mim \
  -o /Users/zc/courses/compiler/products/tile-fuse/results/task28a/pdepth-final-output.mim
```

当前定向 lit 为 closedlm、attn_exec 通过，attention_raise 失败，
见 `depth-targeted.log`；完整重建成功，见 `depth-paused-build.log`。
按任务约定在此停止，保留通用 raising WIP；没有新提交，没有模型计数或计时。
下一步需要决定 matcher 的 continuation 深度预算应如何计量，
或将单个 diamond 拆成更小的有语义 guard 的规则，
但不能在尚未匹配成功时声称调度机制已兑现预测。

### 显式匹配任务链的验证与 FP32 字面量 recipe 限制

第三个阻塞由 `983ba77b0f` 修复，已合入本 worktree 的 `eaac5398de`。
确定性匹配改用显式任务链，只有 Either 分支保存回溯选择，
不再让兄弟节点的数量增加 native continuation 深度；默认 max_depth 仍为 256。
本轮重跑 `pdepth.mim`，输出为 `result(x)=x`，
见 `pdepth-taskchain.mim` 与 `pdepth-taskchain.log`。
至此，首个大型真实 pattern 暴露的三个 P5 问题及其修复均已记录：
闭合调度常量的 alpha 匹配、重复常量出现的独立匹配、兄弟节点与实际嵌套深度的区分。

进一步诊断时，将规则的形状与 scale 临时实例化为测试中的字面量，
正确 attention 根直接匹配返回 Matched、1490 steps，
没有提高预算；见 `taskchain-direct.log`。
图层还补充了高层 dot_product 和 unary 的数据操作 schema，
分别将第六组 tuple 和第四组值声明为输入，未声明全局纯性。
然而 graph rewrite 在构造替换时抛出：
`literal recipes require a Nat or Idx scalar type`。
这与已经修复的匹配深度问题不同。

独立最小复现为结果目录的 `fplit.mim`：

```mim
axm mark: math.F32 → math.F32;
rule replace (x: math.F32): mark x => 0.5:math.F32;
extern lam result (x: math.F32): math.F32 = mark x;
```

使用 `compile.graph_rule_app (replace, mark)` 即退出 1，
见 `fplit.log`；无需 tensor、attention 或 schedule。
`src/mim/pattern/rewrite.cpp` 的 recipe 验证明确只接受 Nat/Idx 的 Literal。
对照 `natlit.mim` 的 Nat 字面量替换成功，
`fpcap.mim` 从目标捕获 FP32 值再返回也成功。
因此限制是 RHS 创建 FP32 字面量，不是所有 FP32 规则均不可用；
捕获已有浮点值是一条已验证的可行路径，不能由此断言所有 attention raising 方案不可表达。

```sh
cd /Users/zc/courses/compiler/mimir-gpm
nice -n 10 build/bin/mim \
  /Users/zc/courses/compiler/products/tile-fuse/results/task28a/fplit.mim \
  -o /Users/zc/courses/compiler/products/tile-fuse/results/task28a/fplit-output.mim -VVVV
```

所有固定形状/scale、core 与 tensor 中的诊断日志均已撤除。
恢复后的通用规则仍未命中，编译本身退出 0，
不能将具体实例的成功匹配等同于通用 raising 已完成。
重建成功；定向 lit 为 closedlm、attn_exec 通过，attention_raise 失败，
见 `restored-generic-build.log`、`restored-generic-targeted.log`、
`restored-generic.log`。
按任务遇阻停止要求保留步骤 1 WIP，不提交、不推送，
未进入 online 规则与模型测量；本轮没有可报告的预测兑现结论。

### FP32 recipe 修复验证与 scale 捕获的归一化边界

第四个阻塞由 `758042d197` 修复：
graph-rule Literal recipe 接受任意闭合标量类型，
回归位于 `lit/rules/fplit.mim`。
本轮重跑独立的 `fplit.mim`，`result` 正确返回 FP32 的 0.5，
见 `fplit-fixed.mim`；没有继续沿用 Nat/Idx 限制的诊断结论。

通用 attention 规则未命中的原因现已隔离到 scalar scale，
不是必须固定所有形状才能匹配：
仅将 scale 参数实例化为 0.5，保留 H/M/S/K/N 五个形状 metavariable，
正确 diamond 的直接匹配为 Matched、2446 steps，
见 `scale-fixed.log`。
该诊断仍有独立的 graph publication 纯性拒绝，
因此不能把直接匹配成功说成完整 raising 成功。
普通依赖形状捕获的独立程序 `generic.mim` 也正确返回输入。

失败来自归一化后的乘法操作数顺序：
`src/mim/plug/math/normalizers.cpp` 的 `fold` 对交换运算使用
`Def::greater` 排序；常量 0.5 在目标中被排到 binder 变量 y 之前，
但规则中尚未绑定的 scale 是 Extract，不是 Lit，
其规范化次序与目标不同。
在当前 attention 诊断中，规则 lambda 的 body 是 `mul(y,scale)`，
目标是 `mul(0.5,y)`；binder 已绑定 y，再拿它匹配 0.5 会失败。
此处没有发生 shape 元变量不一致，也不是依赖类型预算耗尽。

独立复现 `mulcap.mim` 不含 tensor、形状或 QKV view：

```mim
axm mark: [math.F32 → math.F32, math.F32] → math.F32;
rule remove (scale x: math.F32):
    mark (λ y: math.F32: math.F32 =
        math.arith.mul math.mode.contract (y, scale), x) => x;
```

目标将 scale 换为 0.5，保留 mark，不发生替换。
`mulctrl.mim` 将规则和目标的乘法都反写为 `(scale,y)`，
仍被 normalizer 排序回不同方向，仍不替换；
同构的非交换减法 `(y,scale)` 则成功返回 x。
日志与输出分别在 `mulcap.log` / `mulcap-out.mim`、
`mulctrl.log` / `mulctrl-out.mim`。
这说明仅交换 Mim 源代码中的 operand 不是有效修复。

现有设计文档 `graph-pattern-matching-design.zh-CN.md` 与
`graph-pattern-reference-tests.zh-CN.md` 已明确：
native PatternProgram 支持显式 OR 方向，但不提供自动交换律匹配。
直接让 RulePattern 对 math.mul 尝试交换，或对归一化后的任意调用增加其他方向，
会扩展现有 Rule 的匹配语义，不是缺失 capture/budget 等局部实现修复。
另一个选择是通过显式 pattern/规则构造接口表达两个方向；
这同样需要确定接口和归一化交互，不能假装两条反写的源规则已经表达了 OR。
本轮未引入上述设计，也未将整个 attention 识别器藏入 native witness。

所有 scale 固定值、匹配 dump 和 core 调试修改已经撤除；
保留原来的通用 WIP。
重建成功，定向 lit 为 fplit、closedlm、attn_exec 通过，
attention_raise 失败，见 `scale-restored-build.log`、
`scale-restored-targeted.log`。
按新协议仅因涉及规则匹配语义的设计决定而暂停；
没有新提交，没有模型测量，步骤 1–5 仍未完成。

### 用户选定交换律匹配后的机制实现

用户已选定 option A：注册二元算子的 Rule 匹配模交换律进行，
不支持结合律。
机制提交 `fix(pattern): match registered binary operators modulo commutativity`
解决 scale 由 metavariable 变成 Lit 后，normalizer 改变 mul 操作数次序的症状；
根因与修复是将插件现有 `is_commutative` 信息注册到 Driver 的 OpRegistry，
由共享 Rule lowering 生成原顺序优先的 Either，复用任务链回滚与预算。
libmim 不知道 core/math 的具体 axiom，未改变 mode、类型参数或 pipeline 默认值。
独立的 `mulcap.mim` 已正确改写为 x。
`commute.mim`、`grcomm.mim`、`cchoice.mim` 覆盖两个 phase、
嵌套交换、重复捕获、非交换算子、无结合律及原顺序优先/回滚。

继续 attention 时还识别到一个独立的局部 recipe 问题：
从 scalar lambda body 捕获的 Lit 不一定在所选 graph region 的 node 集合中；
graph Rule 把它当作 import，报 `recipe imports a value outside the graph`。
闭合标量 Lit 已可通过第四项修复的 Literal recipe 精确重建，
因此 captured Lit 改用 Literal recipe，不扩大 graph import 的范围。
独立回归 `grlitcap.mim` 使用非交换 sub，避免依赖交换律修复。
此修复与交换律分开提交。

交换律与 captured-Lit 修复后，通用 attention 已进入 graph replacement 验证，
不再停在 shape/scale 的结构匹配阶段。
后续拒绝为 `replacement removes a computation without a purity proof`：
诊断定位到闭合 Church schedule 内部的 `mk(3,0)`。
当前保护逻辑递归检查函数值的 body，
而 tensor.map_reduce_post 的纯性断言只证明其调用，不证明所有函数值参数；
`mk` 是 bound variable，无法作为已知纯 axiom 声明。
诊断 dump 已撤除，尚未放宽该保护条件。

本轮提交前按资源限制串行执行 CTest 与完整 lit：
CTest 249/249；lit 680 passed / 9 unsupported / 1 failed，
唯一失败仍为原有未提交步骤 1 的 `tensor/attention_raise.mim`。
所有新机制回归及已有 pattern suites 通过，未把该 WIP 失败标为 unsupported
或删除检查；步骤 1 仍不作为完成提交。
结果见 `comm-commit-ctest.log`、`comm-commit-full-lit.log`。

交换律机制已提交为 `ceb12bb50deef16839cfb3d80e4a629763203299`。
captured-Lit 的独立提交主题为
`fix(pattern): rebuild captured literals outside graph regions`，
仅修改 recipe 选择并添加 `lit/rules/grlitcap.mim`，
不扩大允许导入的 graph node 集合。

当前设计边界的最小复现保存在结果目录 `schpure.mim`：

```mim
let Schedule = («2; Nat» → Nat) → Nat;
axm mark: [Schedule, Nat] → Nat;
lam schedule (mk: «2; Nat» → Nat): Nat = mk (3, 0);
rule remove (x: Nat): mark (schedule, x) => x;
extern lam result (x: Nat): Nat = mark (schedule, x);
```

使用 `compile.graph_rule_app (remove, mark)` 即退出 1，
报 `replacement removes a computation without a purity proof`，
见 `schpure.log`。它不含 tensor，也不依赖交换律；
使用 monomorphic schedule 只是避免 purity-op 列表的 universe 参数推导干扰。
实际 attention 中的 polymorphic `λ T.λ mk.mk(3,0)` 遭遇同一保护。

`src/mim/pattern/rewrite.cpp` 的现有不变量明确为：
pure operator decoder 仅证明调用的 curry spine，
不证明 function-valued inputs；
递归扫描 closure body 时，未知 bound-function App 必须保留。
因此不能把 `mark` 或 map_reduce_post 声明为 pure，
就悄悄把内部 `mk` 调用从 protected 集合删除。
若要允许此类消除，需要确定闭合 schedule descriptor 的纯性/可丢弃契约，
或增加可验证的 higher-order effect 证明；
这改变纯性证明的边界，而不是遗漏的 literal/capture 实现。
也不能通过给 sdpa 添加仅为保留 closure 的无用参数来绕过检查，
那会修改 op/schedule 表示。
按任务协议在这一设计决定处暂停，保留步骤 1 WIP。
步骤 2–5 尚未执行，没有模型计数或计时。

captured-Lit 提交前再次运行 CTest 与完整 lit：
249/249、680 passed / 9 unsupported / 1 failed，
唯一失败仍为未提交的 attention_raise WIP；
见 `litcap-commit-ctest.log`、`litcap-commit-full-lit.log`。

### lambda 值的纯性边界修复

用户决定 graph-rule 的纯性检查把 lambda 视为值：
移除闭包不执行 body，扫描在所有 Lam 处停止，包括开放闭包。
症状是纯 map_reduce_post 的闭合 schedule 值被当作会执行 `mk(3,0)`；
根因是保护扫描额外递归进入函数值，修复删除该递归边，
不放宽实际求值的未知函数或非纯 axiom App。
机制提交主题为 `fix(pattern): treat lambda values as opaque in purity checks`；
`lit/rules/schpure.mim` 和独立 doctest 的六个子场景覆盖正反例。
定向回归全部通过，通用 `tensor/attention_raise.mim` 首次通过；
这只验证小型非 causal diamond，不代表真实模型、mask 和 online 已完成。

完整验证：CTest 250/250，lit 681 passed / 9 unsupported / 1 failed，
见 `lambda-commit-ctest.log`、`lambda-commit-full-lit.log`。
唯一失败是未经修改的 `lit/error/grbimp.mim`，不是新增回归：
它显式把 nest 注册为 pure，规则把
`nest(λ o.effect(x,o))` 换为 `nest(λ o.x)`，
期望因被删除 lambda body 内的 effect 而报错。
按本轮选定的函数值语义，两个 lambda 都是未执行的值，
该替换成功且退出 0，见 `grbimp-lambda-value.mim`。
因此“所有 lambda body 不参与该检查”和“现有纯性测试保持不变且通过”
在这个具体测试上不能同时满足。
没有为通过旧测试添加特殊例外，也没有修改/跳过旧测试。
机制改动和回归暂未提交；需要决定将该旧测试更新为新语义的成功用例，
并继续用新增的实际执行未知/非纯 App 反例保护效果边界，
或明确一个不同的 lambda 替换语义。
本轮按涉及语义/验收冲突的停止协议暂停，尚未继续步骤 1–5。

### 参数化调用的精确边界

用户澄清前一版跳过所有 lambda body 的语义过宽，撤回该未提交实现：
pure 高阶 axiom 仍可能调用闭包，`grbimp.mim` 暴露的 effect 不能删除。
当前机制提交主题为 `fix(pattern): allow region-local parametric calls in purity checks`：
仍递归扫描 body，只将区域内非 external lambda 绑定参数的调用视为参数化，
不沿 Var/type 边把外部 binder 引入局部集合。
闭合 Church schedule 的 mk 调用因此通过；自由函数和 impure/unregistered axiom
的调用仍拒绝，包括闭包内部调用。
原 `lit/error/grbimp.mim` 未修改；
新增 schpure 与九个 doctest 子场景覆盖 nested/outer 参数、自由参数及纯性负例。
定向结果为 schpure、grbimp 通过，attention_raise 仍有后续纯性拒绝，
不再将上一版跳过 body 后的通过作为当前实现结果。

提交前完整验证：CTest 250/250；
lit 681 passed / 9 unsupported / 1 failed，唯一失败是 attention_raise WIP，
见 `parametric-ctest.log`、`parametric-full-lit.log`。

参数化调用主体修复提交为 `c3a249399cc1352b167289e75a58ad4e145a8dff`。
继续 raising 定位到同一语义的表示遗漏：Mim 的 `fun` 返回 continuation
是 tuple binder 的固定投影，不是裸 Var，
例如 `return_400631` 实际来自局部 lambda 参数组。
补充提交主题为 `fix(pattern): recognize projected local callback parameters`：
沿固定 Extract 链寻找区域内 binder，不穿过任意计算，也不接受自由 binder。
既有自由 tuple 参数反例仍拒绝，新增本地 tuple callback 正例通过；
定向 schpure、grbimp、attention_raise 全部通过。

补充提交前完整验证：CTest 250/250，lit 682 passed / 9 unsupported，
无失败，见 `proj-parametric-ctest.log`、`proj-parametric-full-lit.log`。

补充机制提交为 `2461b052ab`。

## 步骤 1 结果：P5 raising 与普通分解

实现选择 tensor-level `sdpa` / `sdpa_masked` axiom，而不是 native attention matcher：
Mim 中的参数化 Rule 展开完整 dot/scale/稳定 softmax/PV diamond，
P5 负责匹配、依赖 shape 检查和发布，C++ phase 只配置规则、纯性和固定点。
mask 采用原始 score 上的逐点 FP32 加性 bias，支持 causal 的 0/-inf mask；
不把任意非加性 mask 猜成这一语义。
当前自动识别 FP32、mode.contract；其他 mode/精度保留原始图。
lower_tensor 的普通分解逐项恢复原计算，尚未实现 online。
`tensor:attention=0` 或 group=0 关闭新增选择，GPU 不使用 CPU raising。

真实 seq512 attention 和 transformer 的原始 kernel 已各改写一处，
不是根据模型名或变量名匹配：
`model_attention_raised.mim`、`model_transformer_raised.mim` 各含一个 sdpa。
两份输入只追加显式 beta_red/raise/cleanup pipeline 以观察 tensor-level 结果，
保留原始 packed QKV 的 slice/reshape/transpose、bias/residual/LN/MLP 图；
不是完整运行时测量。
`attention_views.mim` 覆盖同构 per-head views 和 residual 使用场景，
`attention_raise.mim` 覆盖普通、mask、错误 axis、late scale、不稳定 softmax。
额外可观察 probabilities reader 被保留，不因替换最终 PV 而丢掉；
该场景可能重复 score 计算，不能据此宣称存储/性能优化。

`sdpa_exec.mim` 对 H=2、M=5、S=7、K=3、N=4，
以确定性输入比较普通分解与标量稳定 softmax 参考：
non-causal 和 causal 都满足 atol=rtol=2e-4。
这里未计时、未插桩；结构验证表明 score/probability 仍是完整张量，
故只确认“raising 保留语义、普通分解仍为二次方中间量”，
步骤 1 的 0.8–1.25 时间预测留待步骤 4，不先报 confirmed。

| 预测 | 当前证据 | 判定 |
|---|---|---|
| 真实 attention/transformer 各 raising 一处 | 两份 seq512 tensor dump 各一处 sdpa | confirmed |
| 普通/加性 causal mask 语义保持 | 标量数值回归，2e-4 容差 | confirmed（小型形状） |
| 普通分解不消除整个 score/probability | 仍恢复标准 dot/softmax/dot | confirmed（结构） |
| FLOP/exp、分配与时间范围不变 | 尚无动态插桩/时间测量 | 尚未测量 |

步骤 1 提交前完整验证：CTest 250/250，lit 684 passed / 9 unsupported，
无失败，见 `step1-ctest-final.log`、`step1-full-lit-final.log`。
前一轮 all build 没有更新 lit staged copy，读到了修改前的 FileCheck 顺序；
显式构建 mim_lit_tests 后重跑完整套件，不以源文件的定向通过代替 staged 验证。

步骤 1 已提交为 `b07049508477f3448cc8df2652481b1d9dfb6ab6`。

### online 实验暴露的静态选择死分支验证问题

online 初稿使用已有 tiled/tile_fold 和普通 dot leaf。
lower_to_mem 在收集 tensor 类型时递归扫描仍未决定的 `if_static`
的 key/静态候选，遇到仅作编译期查询的 blis_candidate 就报
`unbufferizable tensor op`；实际 rewrite 已明确选择 dynamic fallback，
这些 key/静态候选不会被求值。
局部修复主题为 `fix(tensor): skip discarded static selector operands during bufferization`：
类型收集与已有 rewrite 保持一致，只遍历 dynamic 分支和结果类型，
不改变 selector 的决策规则或 pipeline 默认值。
`static_dead.mim` 独立覆盖，online 的 H2/M5/S7/K3/N4、BM2/BK3
随后通过普通及 causal 标量参考（`online-after-static-run.log`）。
online 仍是未提交步骤 2 WIP；此机制修复单独提交。

## 步骤 2：online 规则实现

静态选择器修复提交为 `8314b18af0701c5cc2f0b7a6c78cd687e28dbc14`；
其提交前 CTest 250/250、完整 lit 685 通过、9 unsupported。

新增 `attention_online_rule` 与 masked 对应规则，把步骤 1 construct
改写为 `sdpa_online(mode,masked,BM,BK)`。
后者仍是 tensor-level 复合 op，lowering 才展开成已有 tiled/tile_fold，
没有新增 attention 专用 C++ 循环生成器。
外层 tile=(1,BM,N)，内层 K1 不切、K2 fold 携带 max/sum/output 三元组；
最后一个 K2 块在 fold 外剥离，确保其宽度静态，避免把符号 min 宽度带入 leaf。
例如 S7/BK3 是两次宽3的 fold，随后宽1尾块；S1 则 fold 零次。
规则不接收 S0、N0 或零 tile；直接调用 online op 时以明确诊断拒绝这些参数，
不以 unsigned 下溢或零 tile 除法决定行为。

数值检查复用 `sdpa_aux.cpp` 标量参考：
H2/M5/S7/K3/N4、BM2/BK3，group=1 与 group=0，
普通、causal、大幅值 score、首个 K2 块完全屏蔽但后块有效、整行全屏蔽。
首块全 mask 需要显式令无有效项时 probability=0、carry correction=1；
否则 `-inf-(-inf)` 会使后续有效块也永久变为 NaN。
整行全 mask 的最终 0/0 保持普通 stable softmax 的 NaN，不擅自定义成零。
有限结果仍用 atol=rtol=2e-4，不放宽容差。
`online_single.mim` 另检查 batch1、K2仅1、M尾块和所有维度为1的标量退化。

**结构判定：部分 confirmed，计数/性能待步骤 4。**
规则确实表达了 tile 内分解与异构 carry，并通过以上数值用例；
但当前异构 tuple 的 tile_fold 使用已有 CPS fallback，
不是 direct Arr carry 的原位 buffer 更新。
leaf 仍调用已有 dot scheduling；符号 rows 的 BLIS admission 不成立时，
使用其既有 fallback，不宣称每个 tile 得到了 kind-1 BLIS。
因此 register 常驻、理想 1184 字节工作区与时间区间均尚未确认。
最初动态 min 宽度方案的 group=0 类型错误不再复现：
静态尾块方案的 group=0 数值用例通过。

步骤 2 提交前 CTest 250/250、完整 lit 690 通过、9 unsupported；
日志为 `step2-ctest-final.log` 与 `step2-full-lit-final.log`。
