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

## 步骤 3：显式规则与 group 自动选择

步骤 2 提交为 `aba007b2ea`。
新增 group-aware 的 `tensor.group_attention`，位于 raising 后、通用 lowering 前：
若等到 map_reduce group 阶段才决定，复合 attention 已经被普通分解，时机过晚。
该阶段没有原生 attention matcher，只构造并应用 Mim 的 schedule Rules。
CPU/group=1 时应用，GPU/group=0 不应用；
`-X tensor:attention-online=0` 保留 raising 加普通分解的对照，
`-X tensor:attention=0` 则连 raising 一起禁用。

规则内选择 BK=`vec_width 8`，register budget 复用 `group-registers`（默认32），
BM 取1/2/4中满足 `BM*(max(K,N)+BK+2)<=registers*lanes` 的最大候选，
至少为1；这只是预算模型，不是寄存器分配器的证明。
在 K=N=64 时，默认得到 BM2/BK8；
lanes4 得 BM1/BK4，registers64/lanes8 得 BM4/BK8，
`online_select.mim` 分别检查这些选择及禁用路径。

手写 schedule 显式调用同一 `attention_schedule_rule`，不使用 C++ 专用路径。
`online_schedule.mim` 在自动选择开启和关闭时都调用这条规则，
分别由自动入口或显式入口完成替换；随后 lower_tensor。
两个 realized IR 文件逐字节相同（不只是匹配相同几个 FileCheck 行）：
外层 tiled shape12×512×64/tile1×2×64，内部 CPS K2 fold；
因而 DAG、静态循环参数和 carry 也相同。
独立 `online_select.mim` 不含显式 fallback，防止自动入口失效却被手动规则掩盖。
自动路径的 masked/nonmasked 数值参考由 `online_auto.mim` 覆盖。

真实 seq512 attention、transformer 的 scheduled dump 各恰好一处
`sdpa_online(16,ff,2,8)`，packed QKV views 不妨碍选择。
完整 attention 已生成 LLVM（`model_attention_online.ll`）；
未以此代替运行时模型数值检查或计时。
结构一致的预测 confirmed；动态计数和同进程交替计时留到步骤 4。

新增 annex 改变生成的 enum ID；仅构建 plugins 后，旧 unit binary
会错误地把 phase ID 解释成别的 axiom。
全树重建（包含 unit binaries）后验证通过：
CTest 250/250、完整 lit 693 通过、9 unsupported，
见 `step3-rebuild-all.log`、`step3-ctest-final.log`、`step3-full-lit-final.log`。
这也重新验证了步骤 2 的实现与全部当前 unit consumer 的一致性。

步骤 3 提交为 `7cc84d130b37ab9929e6c12cc3ffd631845c3269`。
后续检查把 masked 空前缀的 correction 单位元明确写成 `1.0:F`：
Mim 的 `1:F` 是位模式，不是浮点数值 1。
该分支当时只乘零 carry，因此既有有限结果用例仍正确，但表达式应与递推定义一致。
修正后的全树重建、CTest 250/250、完整 lit 694 通过、9 unsupported，
见 `early-raise-ctest.log`、`early-raise-full-lit.log`。

## 动态计数揭示的问题及 product carry 修复

第一次 L128 动态实验暴露两点，不能用步骤 3 的局部结构检查代替整条默认 pipeline：
通用 optimize 先行时孤立 core 的 diamond 没有 raising，
所谓 auto 与 current 的 LLVM hash 一致。
实际 exp 为393216，即理论 E196608的两倍；group=0 为196608。
这是既有 grouping 重复 exp 的测得偏离，不把它记成 online 的结果。
ordinary raising 后再分解的 `raised` 与 current hash 也相同。
候选修复将 raising、group_attention 移到通用 optimize 前，
先识别再优化；`online_early.mim` 检查 L128 默认 pipeline 不再申请完整 score 矩阵。

真正选中 online 后，异构 tuple fold 的 CPS fallback 暴露旧 carry 泄漏：
L128 一次调用释放最终输出后仍有6488064字节不可达。
最小 `fold_escape.mim` 不含 attention，3次迭代各申请16字节；
返回结果释放后仍泄漏32字节，LLVM 中没有 free。
复现、计数器、driver 和日志保存在 task28a 结果目录。
因此没有继续大 L 重复运行该版本。

native product-fold 把 immutable Sigma carry 递归分成字段，
tensor 字段各用两块存储 ping-pong，标量字段保持 SSA；
所有 next 字段读同一轮旧状态，不能覆盖另一个字段尚未读完的旧值。
独立 fold 复制 init、结束释放 unused bank，返回 bank 的所有权与普通 tensor 输出相同。
位于已有 sequential tiled 内部且形状静态时，两块 bank 使用既有 tile scratch pool，
在整个 grid 结束统一释放；不逐迭代申请。
非静态或不满足 pool 条件仍使用独立分配，不静默假定可复用。
`product_fold.mim` 用两个互相依赖、形状不同的 tensor 和一个标量，
检查0/1/2/5轮数值、输入值语义与释放输出后零 outstanding allocation。

实现同时修复两个局部边界：
capture_producers 的共享 realization 只能提前生成依赖外部自由变量的 producer，
不能在 tile 参数绑定前生成依赖 tile 参数的计算；
单例轴折叠后，写入 destination 仍须采用 producer 的逻辑 shape/rank。
共享准备越界及逻辑秩失配均由 `online_exec.mim` 的 BM2/M5、BK3/S7 覆盖。
scratch 分配的占位 memory 不能取代 carry 初始化的顺序链：
各字段的 init copy 必须相互依赖，否则后一个 fresh placeholder 会丢掉前一个 copy。
该问题在定向数值验证中发现并修正，而非降低精度要求。

修复后的 L128 auto/manual 动态计数相同：
7次 heap 分配、累计/峰值394272字节，最大分配393216字节为输出；
17次静态大小 stack slot 共2488字节，计入后二者峰值396760字节；
exp221184=E+h·L·ceil(L/8)，释放输出后零泄漏。
工作区为3544字节，**偏离**原理想1184字节预测：
双 bank、普通 tensor leaf 及临时缓冲并没有全部 register 常驻。
仍无完整 S/P，活动 workspace 与 L 无关的结构预测得到支持。
这里 stack 计数按 LLVM alloca 的 payload 字节，不计 ABI padding/编译器 spill；
计数版把 scalar exp intrinsic 替换成带副作用 wrapper，计时使用未插桩库。

在继续 L512/1024/2048 前的预测：相同 BM2/BK8 下，
heap次数7、stack payload2488和额外heap1056应不变，
动态 exp 应分别为3538944/14155776/56623104，
峰值应为输出 `4*h*L*64` 加3544字节。
这些是对新 carry 实现的可证伪预测，不代替原始预测。

product-fold 提交前 CTest250/250，完整 lit695通过、9 unsupported。
池化新增准入仅限 native Sigma fold，保留原 Arr fold 的准入行为：
扩大到所有 fold 曾使 BLIS MLP 的乘法计数多4次，
因此收回该无关扩展，既有 `blis_workloads.mim` 的精确计数保持通过。
见 `product-commit-ctest.log`、`product-commit-lit.log`。

product-fold 修复提交为 `67ce17188f`。
early recognition 不新增全局 beta pass、不改变通用 optimize 的内部顺序，
只把现有 attention 两阶段移到其前；
`online_auto.mim` 改为从真实 decomposed diamond 进入默认 pipeline 并做数值检查，
避免直接输入 sdpa 掩盖未 raising 的问题。
early 提交前再次 CTest250/250、完整 lit695通过、9 unsupported，
见 `early-commit-ctest.log`、`early-commit-lit.log`。

## 步骤 4：四种长度的动态验证

计数与计时版本为 `18aab8fc9ec0c8cf4ddb8bcea85448d6253ba1c3`。
所有文件均位于上述 task28a 结果目录：
`measure_core.py`、`core_results.json`、`core-matrix.log` 保存孤立 core；
`measure_models.py`、`models_<L>_<variant>_18aab8fc9e*/results.json`
保存完整 attention/transformer。
`current` 禁用 attention raising，`raised` 只 raising 后普通分解，
`auto` 是 BM2/BK8 online，`manual` 显式应用相同规则。
计时使用未插桩的 LLVM，以 clang `-O3` 编译：
core 为 `-mcpu=native`，完整模型为配置中的 `-mcpu=apple-m2`；
计数库不用于延迟比较。

复用已完整验证的本地 Debug 编译器，而非 gemm-bench 默认远端 fresh Release 构建。
这是工具链准备方式的偏离，不把 host 编译器的 Debug 开销算入 kernel 时间；
生成的 native kernel 仍经过上述相同 clang 优化。
机器为 Apple M2 Pro；所有运行单线程、预热3次、3组样本取中位数，
每组至少运行200ms，重任务串行并使用 nice。
macOS 没有绑核或隔离核心，不能解释小幅跨进程波动。

### 孤立 core 的时间与正确性

单位 ms。core 在同一进程内交替正反 backend 顺序；
eager 明确执行 matmul、stable softmax、matmul，不调用 fused SDPA。

| L | group=0 | current | raised | auto | manual | eager |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 1.379 | 1.473 | 1.479 | 5.579 | 5.638 | 0.366 |
| 512 | 23.934 | 24.350 | 24.213 | 90.982 | 90.531 | 5.519 |
| 1024 | 115.304 | 107.399 | 107.889 | 386.252 | 385.967 | 26.607 |
| 2048 | 444.123 | 409.647 | 463.656 | 1495.800 | 1524.200 | 92.836 |

所有有限输出满足原定 atol=rtol=2e-4，最大绝对误差小于1e-7。
auto/manual 时间比依次为0.9895/1.0050/1.0007/0.9814，
同进程相差不超过2%，符合原10%预测。
每个 L 的 raised/current 原始 LLVM SHA256 完全相同；
L2048 的时间差不能归因为 raising 改变了执行结构。

auto/current 延迟比为3.79/3.74/3.60/3.65，
落在最初宽范围0.7–5内，但没有获得性能收益。
按 QK+PV 理论 FLOP 计算只有8.34–9.02 GFLOPS，
明显低于“leaf 同等有效”条件下40–100 GFLOPS的假设。
小 BK、carry 读写和 fallback leaf 使该条件不成立；
这验证了规则能实现 online 结构，不验证现有 leaf 足以实现 IREE 的吞吐。

### 第一轮计数：bufferized LLVM 的动态上界

**本小节不是最终 native 计数。**
第一轮先把 malloc/exp 改为带副作用 wrapper，再交给 clang 优化。
后续交叉检查发现这会阻止部分分配消除和 exp CSE；
因此保留这些数值作为 bufferized IR 的动态证据，
但不能用它们声称未插桩 native 确实有同样的冗余。
后面的“优化后计数”才用于最终预测判定。

以下 peak 均不计调用者已持有的 Q/K/V。
group=0/current 两列为动态 heap peak；online 列另包含原始 LLVM alloca payload。
没有把 payload 当作 ABI 对齐后的真实栈帧大小，也不包含 LLVM 后端新增 spill。
每次返回后释放最终输出，所有路径 outstanding heap 字节均为0。

| L | group=0 exp | current exp | auto/manual exp | group=0 heap peak | current heap peak | online heap+stack peak |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 196608 | 393216 | 221184 | 2758656 | 3676160 | 396760 |
| 512 | 3145728 | 6291456 | 3538944 | 39346176 | 53108736 | 1576408 |
| 1024 | 12582912 | 12582912 | 14155776 | 154189824 | 158466048 | 3149272 |
| 2048 | 50331648 | 50331648 | 56623104 | 610369536 | 627179520 | 6295000 |

online 四个 L 均为7次 heap 分配，累计字节等于 heap peak，
即输出 `4*12*L*64` 加1056字节；最大单笔申请就是输出。
另有17个 stack slot，payload 总计2488字节。
因此活动峰值精确为输出加3544字节，auto/manual 逐项相同。
原始 LLVM 中没有完整 S/P；修复后的动态计数也没有二次方活动中间量。
分配次数、额外 heap 和 stack 对 L 不变的后续预测全部 confirmed；
原先理想1184字节的 register-resident 工作区预测仍是 deviates，不能回填预测。

online exp 恰为 `E+12*L*ceil(L/8)=1.125E`，包括首块 correction，
没有把被乘零的首块 exp 假设成已经消除。
current 的插桩 IR 在128/512为2E，在1024/2048为E；
这提示 lowering 中存在 shape-dependent 的重复 exp，但不证明 native 也重复。
优化后交叉检查在 L128 将2E消为E，不能把前一轮观察写成最终运行时缺陷。
插桩把 scalar exp intrinsic 替换为带副作用 wrapper，
遇到未处理的向量 exp 或动态/未知 alloca 会明确失败，不默认为零。

### 完整 attention/transformer 时间

单位 ms。每个变体的 gemm-bench runner 都与自身 eager 交替计时；
下表 eager 取 auto 那次运行的配对基线，其余 eager 样本仍保留在各 results.json。
完整模型 eager 是 native ATen 循环，使用元数据注明的
OpenBLAS 0.3.34 interposed BLAS，不是 core 的 Python eager 调用口径；
不把两张表的 eager 差值当作某个调度的收益。

| 模型 | L | group=0 | current | auto | manual | eager |
|---|---:|---:|---:|---:|---:|---:|
| attention | 128 | 14.953 | 13.500 | 347.763 | 346.119 | 7.229 |
| attention | 512 | 105.585 | 75.374 | 1509.945 | 1470.987 | 38.891 |
| attention | 1024 | 371.026 | 201.875 | 3184.325 | 3152.465 | 101.249 |
| attention | 2048 | 1269.349 | 620.372 | 6837.130 | 7025.229 | 304.140 |
| transformer | 128 | 47.181 | 28.050 | 355.855 | 382.144 | 20.335 |
| transformer | 512 | 226.601 | 124.813 | 1571.006 | 1529.136 | 85.494 |
| transformer | 1024 | 625.428 | 323.711 | 3323.390 | 3293.429 | 203.700 |
| transformer | 2048 | 1760.238 | 873.887 | 7100.819 | 7574.678 | 501.455 |

汇总逐项检查 status，而非只看 runner 的进程退出码。
成功运行的数值检查均满足2e-4容差，attention 最大绝对误差小于1.72e-7，
transformer 不超过3.82e-6。
完整模型 auto/current 时间比：attention 为25.76/20.03/15.77/11.02，
transformer 为12.69/12.59/10.27/8.13。
这不能由孤立 core 的3.6–3.8倍 slowdown 单独解释，也不能称为“仅仅 leaf 较慢”。

运行中并发集成把主 worktree HEAD 从18aab8fc9e推进到了ffac206211。
版本 guard 拒绝了当时尚未开始的 L2048 current/manual；
即使 runner 退出码为0，两份失败结果也没有作为有效测量使用。
没有回退或覆盖他人提交，而在结果目录建立 detached 的
`compiler-18aab8fc9e`，恢复相同 host compiler、SDK、Debug/checks 配置，
完整构建并通过 CTest250/250、lit695通过/9 unsupported，再补齐缺项。
独立构建不启用 Python bindings，不影响生成的 kernel。
补测采用新的 `_r2` 目录，保留失败尝试供追溯。

随后 `verify_compiler_provenance.py` 用固定 SHA 的编译器重新生成四个 L、
两个模型和所有四种 MimIR 对照的32份 LLVM。
`compiler_provenance.json` 中32份原始/重生成 SHA256 全部相同，
因此并发 HEAD 变化没有使表中的 kernel 混成不同版本。
这验证的是生成程序的一致性，不证明运行时没有系统噪声。
完整模型 auto/manual 的严格同进程交替比较另列，不用上面的跨进程表代替。

### 手动与自动的同进程模型比较

`compare_model_schedules.py` 在一个进程内加载同版本的 auto/manual 库，
使用相同输入和参数，分别先做数值检查与3次预热，再交替执行3组。
仍为单线程，每组至少200ms，包含调用和释放输出；
此处不重新编译或使用计数库。结果为 `model_schedule_comparison.json`。

| 模型 | L | auto ms | manual ms | auto/manual |
|---|---:|---:|---:|---:|
| attention | 128 | 343.612 | 347.074 | 0.99002 |
| attention | 512 | 1442.900 | 1441.470 | 1.00099 |
| attention | 1024 | 3085.653 | 3081.542 | 1.00133 |
| attention | 2048 | 6875.997 | 6898.036 | 0.99681 |
| transformer | 128 | 356.573 | 358.149 | 0.99560 |
| transformer | 512 | 1496.217 | 1497.126 | 0.99939 |
| transformer | 1024 | 3177.394 | 3193.835 | 0.99485 |
| transformer | 2048 | 7091.937 | 7087.751 | 1.00059 |

8项均相差不超过1%，原10%预测 confirmed。
这排除了“自动入口选择了另一套 online 实现”作为 slowdown 的解释，
但不能排除两条路径共同存在的低效。

### 优化后计数：先排除插桩观察者效应

对 L128 完整 attention 的诊断预测单独保存为
`native_counter_crosscheck_prediction.json`，之后才做交叉检查。
先对**未插桩** LLVM 做正常 clang `-O3` 优化、输出 LLVM，
再替换 allocator/exp 并插入动态乘法计数。
移除原 intrinsic/allocator 的 memory 属性，避免 wrapper 被错误当成无副作用；
向量 exp 按 lane 计数，FMA 计一个乘项，未知 alloca 明确失败。
`crosscheck_native_counts.py` 与 `native_counter_crosscheck.json` 保存最初两例。

结果推翻了诊断中“current native 仍执行2E次 exp”的预测：
current 的393216降至196608，分配也从14116次/峰值12982272
降为13964次/峰值4724736字节。
auto 的 exp221184和乘法555442176则保持，
分配从9285次/峰值4718592降为9255次/峰值1968888。
因此“exp 在 native 重复”这一判断撤回；
“online 令完整模型额外执行一次 QKV projection”仍有优化后证据。
不能用插桩抑制 CSE/分配消除来解释未插桩库的时间。

最终用 `measure_native_counts.py` 对所有 core/完整模型重做该口径，
结果另存 `native_counts.json`，不覆盖前一轮 IR 计数。
这里的 heap 是优化后程序实际经过 wrapper 的申请/释放；
stack 仍只核算 LLVM 显式 alloca 容量，不计 ABI padding、slot 重用和后端 spill，
不是测得的整个进程 RSS 或物理 native 栈峰值。

#### 优化后的 core 结果

52项计数全部完成，所有输出满足原容差，最大绝对误差小于3.34e-6，
释放输出后均无 heap 泄漏；每个模型/长度的 auto/manual 各字段完全相同。
core 的 raised/current 计数也完全相同。
普通 group=0/current 在四个 L 的 exp 都为E，online 都为1.125E。
下表最后一列为 heap peak 加显式 alloca 容量，非物理栈测量。

| L | current exp | online exp | group=0 heap peak | current heap peak | online heap+alloca |
|---|---:|---:|---:|---:|---:|
| 128 | 196608 | 221184 | 2758656 | 2857984 | 395808 |
| 512 | 3145728 | 3538944 | 39346176 | 40461312 | 1575456 |
| 1024 | 12582912 | 14155776 | 154189824 | 158401536 | 3148320 |
| 2048 | 50331648 | 56623104 | 610369536 | 627115008 | 6294048 |

online 仍是7次 heap 申请，累计=峰值=输出加1056字节；
优化后只剩3个显式 alloca，合计1536字节。
因此该口径下的固定额外容量为2592字节，而非前一轮3544字节。
原始1184字节理想工作区预测仍 deviates：
carry 的双 bank 和 leaf 临时值没有全部留在寄存器。
“无完整 S/P、额外空间不随 L 增长”则 confirmed。
这里只验证静态输入形状、运行时 tile grid 的非 causal 实验；
外部输入或程序自身生成的完整 causal mask 不能从存储账目中隐去，
本矩阵没有验证完整 causal 模型也满足同一峰值。

#### 优化后的完整模型存储与计算

均不计调用者持有的输入/参数；本矩阵完整模型无剩余显式 LLVM alloca，
不意味着最终机器码没有 spill。字节为十进制整数。

| 模型 | L | group=0 heap peak | current heap peak | online heap peak | online 申请次数 | online 累计字节 |
|---|---:|---:|---:|---:|---:|---:|
| attention | 128 | 5505536 | 4724736 | 1968888 | 9255 | 23595768 |
| attention | 512 | 50333696 | 46752768 | 7867128 | 36903 | 87296760 |
| attention | 1024 | 176164864 | 170984448 | 15731448 | 73767 | 172231416 |
| attention | 2048 | 654319616 | 652280832 | 31460088 | 147495 | 342100728 |
| transformer | 128 | 8651264 | 6684672 | 3935480 | 9424 | 47977208 |
| transformer | 512 | 62916608 | 54592512 | 15733496 | 37072 | 128199416 |
| transformer | 1024 | 201330688 | 186713088 | 31464184 | 73936 | 235162360 |
| transformer | 2048 | 704651264 | 683746304 | 62925560 | 147664 | 449088248 |

online 的 peak 线性增长，完整 scores/probabilities 的二次方空间被消除；
但不应把 core 的固定7次申请套到完整模型。
这组形状下完整 attention/transformer 分别为 `72*L+39`、`72*L+208` 次申请。
上下文中的 producer/leaf realization 仍影响分配，不能只看局部 op 的循环形状。
两个完整模型的普通 exp 均为E，online 为1.125E，与 core 一致。

下表计乘法/FMA的乘项，不把 layer norm 或 scale 的乘法误记成 GEMM FLOP。
current 的主要 GEMM 乘项预测为
`4*L*768² + 2*12*L²*64`，
transformer 再加 `2*L*768*3072`；
表中还包含 elementwise 算术。

| L | attention current | attention online | transformer current | transformer online |
|---|---:|---:|---:|---:|
| 128 | 327352320 | 555442176 | 931725312 | 1159815168 |
| 512 | 1613758464 | 2545287168 | 4031250432 | 4962779136 |
| 1024 | 4039114752 | 5953290240 | 8874098688 | 10788274176 |
| 2048 | 11324620800 | 15357444096 | 20994588672 | 25027411968 |

两种模型、四种长度的差值都**精确**为：

```text
online-current = 3*L*768² + 12*L*ceil(L/8)*(64+1)
```

后项是原来已经预测的 output/row-sum carry rescale；
前项恰为额外一次完整 QKV projection，违反“主要 GEMM 量不变”的预测。
L128 的未插桩优化后 LLVM 中仍有两组128×2304×768的 projection nest；
原始 LLVM 两组读取相同 x/p0，并写入相同 QKV buffer，分别服务不同 views。
故这里不是凭时间猜重算，也不是插桩抑制优化产生的假象。
该诊断假设写在 `model_count_predictions.json` 后才测乘法；
它是看到时间偏离后的归因实验，不冒充最初的先验预测。

`model128-trace.log` 的 kind-1 日志只给输出 projection 选择了 BLIS，
QKV 没有同样的请求；优化后仍是普通 scalar reduction nest。
`src/mim/plug/tensor/phase/group.cpp` 的 `Group::start`
把具有 region occurrence 的候选预先 claimed，保守地不自动调度它们。
这与 online 分解后跨 tile region 的 QKV producer 未获准入相符。
此外 bufferization 并未保持跨这些 views 的 producer 单次 realization。
额外工作量与失去 BLIS 是直接证据；没有分别计时 projection，
不能把总 slowdown 精确分摊成它们各占多少。
完整模型 online 的分配次数反而少于 current，不能把25倍 slowdown
简单归因为“malloc 更多”。

此处不按 L/模型名加特例，也不靠改 iteration budget 隐藏偏离。
要恢复预测，需要让跨 region producer 的 placement、group 准入和
单次 realization 保持一致，而不是仅在 attention op 内再换一个 BK。

### 逐步骤判定

| 步骤 | 预测 | 测得 | 判定 |
|---|---|---|---|
| 1 raising | 真实 diamond 命中，near miss 不误吸收，普通分解保持语义 | 两模型命中；正反例及标量参考通过；core raised/current LLVM 完全相同，时间在0.8–1.25内 | confirmed（完整模型未另测 raised-only 时间） |
| 2 online | tile 内 recurrence，主要 GEMM 不变，exp增加12.5% | core 符合；完整模型额外一次 QKV projection | core confirmed，完整模型 deviates |
| 2 存储 | 消除完整 S/P，理想 workspace1184字节 | core 显式容量输出+2592；完整模型峰值线性 | 空间阶数 confirmed，理想常数 deviates |
| 3 入口等价 | 手动/自动同结构、同计数，同进程时间相差≤10% | realized IR一致，52项中配对计数相同；core≤2%，两完整模型≤1% | confirmed |
| 4 时间 | leaf 同等有效时40–100 GFLOPS；初始 core允许0.7–5倍 | core8.34–9.02 GFLOPS且慢3.6–3.8倍；完整模型慢8.1–25.8倍 | core宽范围内，leaf假设及端到端效果 deviates |
| 4 计数方法 | 插桩反映实际执行 | pre-clang插桩阻止CSE/分配消除；post-clang复测纠正 | 第一轮方法 deviates，最终采用优化后口径 |

**结论：**P5规则、tile/fold和group入口能正确表达并自动选择这组 online 算法，
但当前跨 region producer 调度/realization 未兑现整模型的工作量预测。
空间改善不是性能成功的替代指标；本实验不支持把该策略称为可用的默认 CPU 性能策略。

步骤4报告提交前，在已经集成其他 P5 改动的主 worktree 完整重建，
CTest257/257、完整 lit700通过/9 unsupported，
见 `step4-build.log`、`step4-ctest.log`、`step4-lit.log`。
这些集成验证不替换性能表的固定18aab8fc9e版本标识。
