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
