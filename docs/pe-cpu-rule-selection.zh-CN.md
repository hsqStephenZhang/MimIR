# 按算子选择调度规则：现状、业界做法与 MimIR 方案

本文回答一个问题：`lower_to_mem` 里“这个算子能不能走 PE、走哪棵树”的判断，怎样做得更完善，让不同算子按各自的规则选择，并尽量放到 Mim 里做。
调研对象是 IREE（`lowering_config`）、TVM DLight 和 PyTorch Inductor，三者都读的是源码（版本见各节）。

---

## 1. 现状

### 1.1 判断分散在三处

| 位置 | 做什么 | 性质 |
| :--- | :--- | :--- |
| `normalizers.cpp::pe_cpu_plan`（约 80 行） | 判定“2 输入、1 个归约轴、BLIS 能覆盖”的收缩，并抽取 `dep`、`a`、`unit_b`、`Aff` | 判定和抽取事实混在一起 |
| `normalizers.cpp::pe_generic_plan`（约 70 行） | 判定“仿射读映射、输出是并行轴的排列”的任意 map-reduce，并抽取 `red_unit`、`out_unit`、`wdim`、`rdim`、`outer`、`merge`、`lanes` | 同上，而且 `outer` 这类事实只服务于某一个策略 |
| `lower_to_mem.cpp::lower_map_reduce` | 固定顺序：`lower_pe_cpu` → `lower_pe_generic` → btensor 普通循环 | 优先级写死在代码里 |
| `tensor.mim::conv_impl` | 构造 im2col / flat / direct / `conv_schedule` 几种候选，通过 `pe_cpu_candidate` 问 C++“PE 能不能接” | 算子级选择在 Mim，判定在 C++ |
| `lower_to_mem.cpp::fuse_producer` | 是否融合 producer：使用次数 ≤ 2、收缩只经线性行融合 | 融合策略写在 C++ |

策略本身（`cpu_tree`、`generic_tree`）已经在 Mim 里，但“谁来用哪个策略”和“策略需要的事实”都在 C++。

### 1.2 具体问题

1. **规则不可扩展**：加一种算子模式（例如 stride-2 卷积的外积块、attention）要同时改 C++ 的 plan 结构体、判定函数和 Mim 策略签名。
2. **事实和策略耦合**：`PeGeneric::outer`、`PeCpu::unit_b` 只为某一个策略而存在，却由通用的判定函数计算。
3. **没有“拒绝”协议**：策略在 Mim 里只能产出一棵树；树不合法时 `refly.check` 直接报错，而不是换下一条规则。
4. **没有覆盖入口**：除了 `-X tensor:cpu-levels` 和若干 `cpu-*` 旋钮，用户无法针对某个算子指定树或规则。
5. **决策不可单独测试**：只能通过 `-VVV` 的树打印间接验证。

---

## 2. 三家做法

### 2.1 IREE（main，commit `87532f1`）

- **入口**：`LLVMCPUSelectLoweringStrategyPass` → `initCPULaunchConfig`（`KernelDispatch.cpp`）。
  函数上已有 `translation_info` 就直接返回，这就是用户配置优先的实现方式。
- **根算子**：`getRootOperation`（`CPUUtils.cpp`）按“带归约的 Linalg 算子 > 其他可分块算子 > 逐元素 > 纯重排”的固定排名选一个根；融合进来的其他算子的配置由根的配置推导（`MultiLoweringConfigGenerator`）。
- **分派**：`setRootConfigImpl` 按严格优先级尝试：具体算子类型（attention、fft、gather、pack、pad、mmt4d）→ 结构识别（`is2DConvOp`、`isaContractionOpInterface` 加“恰好一个归约轴且在最内层”）→ `GenericOp` 子链（转置 → 逐元素 → 默认，`failure()` 表示“不归我，试下一个”）→ 通用 `TilingInterface` → `Default` 流水线。
- **配置是 IR 里的数据**：`#iree_cpu.lowering_config` 是一个按名字分级的字典（`distribution`、`cache_parallel`、`cache_reduction`、`vector_common_parallel`、`vector_reduction`、`vector_inner_parallel`），`translation_info` 选一条流水线（`DoubleTilingExpert`、`Mmt4dTilingExpert`、`ConvTileAndDecomposeExpert` 等）。
  流水线只读属性，按出现了哪些级别添加分块 pass。
- **覆盖**：源码里的 `#iree_codegen.compilation_info` 或 tuning spec（transform dialect，`foreach_match` 匹配后只做 `annotate compilation_info`），两条路径都汇入同一个“用户配置”通道。
- **校验**：每种流水线有自己的 verifier（例如并行级别只能切并行轴、归约级别只能切归约轴、interchange 必须是排列），对启发式和用户配置一视同仁。
- **目标参数**：统一从 target descriptor 读（`native_vector_size`、CPU features、`ukernels`）；data tiling 和 ukernel 的选择在单独的阶段做。

### 2.2 TVM DLight（main，commit `1091f2f`，现位于 `python/tvm/s_tir/dlight/`）

- **规则接口**：`ScheduleRule.apply(func, target, tunable) -> None | Schedule`，返回 `None` 表示不适用。
- **驱动**：`ApplyDefaultSchedule(*rules)` 对每个函数线性尝试规则，第一个不返回 `None` 的胜出，然后打上 `tirx.is_scheduled` 标记；后续阶段都跳过已标记的函数。
  GPU 默认顺序是 `Matmul, GEMV, Reduction, GeneralReduction, Fallback`，`Fallback` 保证总有结果。
- **先规范化再匹配**：`normalize_function` 给每个 block 算出迭代类型串（如 `"SSR"`），规则只看这种摘要。
  `Matmul` 按“每个迭代变量出现在哪些操作数里”把迭代分类成 S/I/J/K，再把任何类 GEMM 的块规范化成 `C[S,I,J] += A[S,I,K]·B[S,J,K]`，一个模板服务所有变体。
- **分层**：图级库匹配（cuBLAS 等）→ MetaSchedule 调优数据库 → DLight 规则表 → Fallback，每层都可以拒绝。
- **弱点**：按 block 个数、名字子串匹配，比较脆弱；规则在最后几个检查之前就修改了 schedule，有些检查是 `assert`，会崩溃而不是拒绝；`tunable` 接口在仓库里没有任何规则用到；CPU 规则很少（只有 GEMV 和一类归约），没有 CPU GEMM。

### 2.3 PyTorch Inductor（torch 2.12）

- **算子级注册表**：`lowering.py` 的 `lowerings` 字典，键是 aten 算子；`user_lowerings` 优先，返回 `None` 就退回内置实现；没注册的算子隐式退回 ATen（`fallback_handler`）。
  分解（decomposition）可以按形状返回 `NotImplemented` 来拒绝。
- **候选 + 选择器**：`tuned_mm` 等函数按谓词（`use_cpp_gemm_template`、`use_triton_template`，都在 `utils.py`）往候选列表里加 ATen、Triton、C++ 模板，再交给 `autotune_select_algorithm`；只有一个候选时不做测量。
- **CPU GEMM 模板**：`CppGemmTemplate` 三级分块（寄存器块来自微内核表、线程块做因子搜索、cache 块按 L1/L2 大小），微内核由 `register_micro_gemm` 的声明式表按 ISA 筛选，再按（ISA 等级、整除性、占用率、寄存器块大小）的字典序打分。
- **逐元素/归约**：`TilingSelect.select_tiling` 按步长选向量化轴，有一堆阈值；不支持向量化的单个算子会被局部标量化，不拖累整个内核。
- **启发式和正确性分开**：`InductorChoices` 只放性能启发式，可整体替换；融合合法性在 scheduler 的 `can_fuse`。
- **弱点**：谓词散落在多个文件里，每个 `tuned_*` 都重复一串 `if use_X: add choices`；C++ GEMM 模板只有在 `max_autotune` 加权重冻结时才可能被选中；模板是字符串，对 IR 不透明；CPU 上没有卷积模板。

### 2.4 对比

| 维度 | IREE | DLight | Inductor | MimIR 现状 |
| :--- | :--- | :--- | :--- | :--- |
| 规则单位 | 根算子 + 结构识别函数 | `ScheduleRule` 对象 | 每个 aten 算子一个 lowering | 两个 C++ plan 函数 |
| 选择方式 | 固定优先级，`failure()` 换下一个 | 有序列表，`None` 换下一个 | 谓词收集候选，选择器挑一个 | 固定两步，再退回 btensor |
| 识别依据 | 接口 + 结构（`isaContraction…`） | 规范化后的迭代分类 | 算子名 + 形状谓词 | C++ 里探测访问映射 |
| 决策产物 | IR 属性（`lowering_config`） | 改写后的 TIR | IR 节点 / 模板 | 一棵 `SchedTree`（Mim 项） |
| 用户覆盖 | `compilation_info`、tuning spec | 调优数据库 | `user_lowerings`、`InductorChoices` | 几个 `-X` 旋钮 |
| 合法性 | 每条流水线一个 verifier | 无统一机制 | 分散在谓词里 | 解释器里的 `refly.check`（报错） |
| 兜底 | `Default` 流水线 | `Fallback` 规则 | ATen 外部调用 | btensor 普通循环 |

---

## 3. 该借鉴什么

1. **有序规则表 + 明确的“拒绝”**（DLight、IREE 子链、Inductor 的 `None`/`NotImplemented`）：策略就是列表顺序，加规则不改驱动。
2. **先规范化、再匹配**（DLight 的 S/I/J/K 分类，IREE 的结构识别）：规则看的是统一的事实摘要，不各自去探测 IR。
3. **决策是数据**（IREE）：我们的 `SchedTree` 本来就是比 `lowering_config` 更强的数据形式，缺的是“谁产出它”的统一入口和可替换性。
4. **规则不可信，校验器可信**（IREE 的 verifier）：规则可以随便产出树，由解释器的检查决定是否接受。我们已经有完整的检查，只差把“报错”变成“拒绝并换下一条”的选项。
5. **用户覆盖走同一通道**（IREE）：覆盖只是“在规则表前面插一条规则”或“给某个算子直接指定树”。
6. **目标参数集中读取**（三家都是）：我们的 `cpu_knobs` 已经是这个角色。

不该照搬的：DLight 按名字子串和 block 个数匹配、在拒绝前修改状态；Inductor 把谓词散落在多处、用字符串模板；IREE 的流水线枚举（我们只有一个解释器，不需要多条流水线）。

---

## 4. 方案：规则表放进 Mim

### 4.1 分层

```
C++  reflect      map_reduce_post ──► OpFacts（只描述算子，不含任何策略）
Mim  rules        OpFacts ──► 依次尝试规则 ──► Option SchedTree
Mim  verifier     树在 OpFacts 上是否合法（静态，不生成代码）──► 不合法就当作拒绝
Mim  interpreter  run_levels_at（现有的 refly.check 作为最后一道保险）
C++  fallback     全部拒绝 ──► btensor 普通循环
```

访问映射和规约的 monoid 都已是一阶数据：impl 直接构造 `Aff`，规约用 `tensor.fold_of (op, h)` 声明 `op`。
C++ 不再检查 lambda 的函数体，只做 `aff_canon`（把字面量值内联进行、运行时值留作符号）和 `monoid_of` 对原语 `op` 的判断。

### 4.2 `OpFacts`：一次反射，所有规则共用

```
anx nom OpFacts (N Rn: Nat) = [
    Ro:     Nat,                    // 并行轴个数（前 Ro 个轴）
    sizes:  «Rn; Nat», dyn: «Rn; Bool»,
    maps:   «i: N; Aff (Rn, Rs#i)», // 每个输入的访问映射（数据形式）
    out:    «Ro; Nat»,              // 输出第 k 维对应哪个循环轴（排列）
    bytes:  Nat,                    // 输出元素大小
    merge:  Option ([To, To] → To), // 组合子可重结合时的二元运算
    has_epi: Bool,                  // 是否有需要在 Leaf 里做的 epilogue
]
```

`dep`、`unit`、`wdim`、`rdim`、`outer`、`red_unit`、`out_unit` 全部改成 Mim 里基于 `maps` 的派生函数（`aff_reads`、`aff_unit`、`aff_coef` 已经在 Mim 里）。
这样 `PeCpu`、`PeGeneric` 两个结构体和 `pe_cpu_plan`、`pe_generic_plan` 的大部分代码可以删掉。

在此之上加一个 DLight 式的迭代分类，供规则做结构识别：

```
// 每个循环轴：是否并行；被哪些输入读、是否单位步长
lam loop_kinds {N Rn: Nat} (f: OpFacts (N, Rn)): «Rn; LoopKind»
// 收缩识别：两个输入，恰好一个归约轴 K，存在只被 A 读的并行轴 I、只被 B 读的并行轴 J
lam as_contraction {N Rn: Nat} (f: OpFacts (N, Rn)): Option Contraction
```

### 4.3 规则与规则表

```
// 一条规则：适用就给出一棵树，否则 None
let Rule (N Rn: Nat) = OpFacts (N, Rn) → Option (SchedTree (N, Rn));

// 默认 CPU 规则表，顺序即优先级；最后一条总是适用
anx lam cpu_rules (N Rn: Nat): «k; Rule (N, Rn)» = (
    rule_user,        // 用户规则（见 4.5），默认总是拒绝
    rule_blis,        // as_contraction 成功且 BLIS 能覆盖 → cpu_tree
    rule_conv_outer,  // 两个最内输出轴是外积（直接卷积）→ 外积寄存器块
    rule_transpose,   // 输出和输入在不同轴上连续 → 双轴切块
    rule_row,         // 只有最后一个输出轴连续或窗口很短 → 行向量化
    rule_reduce,      // 连续的归约轴且有 merge → Reduce + Leaf
    rule_leaf,        // 兜底：单个 Leaf
);

// 驱动：第一条既适用、又通过静态校验的规则胜出
anx lam select_tree {N Rn: Nat} (rules: «k; Rule (N, Rn)», f: OpFacts (N, Rn)): Option (SchedTree (N, Rn))
```

现在的 `generic_tree` 是把 `row`、`split`（rfactor）、转置、外积、融合 band 几个分支用 `core.select` 拼在一起的。
拆成独立规则后，每条规则只关心一种模式；融合 band（`Stage` + `Alloc` 包装）是对任何规则结果都适用的后处理，作为规则表外的一层 `wrap_fused`。

### 4.4 静态校验：把“报错”变成“拒绝”

现在的检查都在 `exec_*` 里，只能在展开时报错。
仿照 `levels_need`（`Alloc` 用的静态容量计算），加一个对树的静态遍历 `sched_valid (f, t): Bool`，只追踪 `axes` 的 `cap`/`dyn`/`part`/`red` 和每个 slot 的状态（是否已 Stage、是否被 Pack、`unit`），复用现有检查的条件。

- `select_tree` 用它过滤：不合法的树等于规则拒绝，自动尝试下一条。
- `run_levels_at` 里的 `refly.check` 保持不变，作为最后一道保险；`-X refly:strict=1` 仍然把它变成编译错误。
- 好处是规则可以写得激进一些（例如“先试外积块”），不合法时安全退回，这正是 DLight 缺的东西。

`sched_valid` 只要和 `exec_*` 的检查条件一致即可。
为避免两份条件不同步，可以把每条检查的条件提成一个小函数（例如 `split_ok (a, ts, mode)`），`exec_*` 和 `sched_valid` 共用。

### 4.5 用户覆盖

按代价从低到高：

1. **`-X tensor:rules=blis,row,leaf`**：重排或关闭默认规则，替代现在零散的 `pe-cpu=0`、`pe-generic=0`、`cpu-conv=1`。
2. **用户规则文件**：用户写一个 `.mim`，定义 `anx lam user_rule`，通过 `-X tensor:user-rules=<file>` 或 import 加载，放在规则表第一位。
   这对应 IREE 的 tuning spec：匹配条件 + 常量树。
3. **逐算子指定**：在源码里用一个包装轴 `tensor.with_sched tree op`，`lower_to_mem` 看到就直接用这棵树（仍然经过校验）。
   这对应 IREE 的 `compilation_info`。

### 4.6 算子级选择（卷积等）

`conv_impl` 里 im2col / direct / `conv_schedule` 的选择是“候选 + 选择器”，和 Inductor 的 `tuned_mm` 同构。
建议同样做成有序候选表：每个候选是“改写成哪种 map-reduce”，选择器对每个候选跑 4.3 的规则表，第一个有规则接住的候选胜出。
这样 stride-2 卷积缺外积规则时，会自然退回 im2col，而不是走到很慢的通用叶子。
现在的 `pe_cpu_candidate` 轴就变成“对候选跑一遍 `select_tree`”。

### 4.7 C++ 剩下什么

- `reflect_facts(app) -> OpFacts`：`aff_canon`、`fold_of` 的 `op` 是否 monoid、输出排列、元素大小。
- `lower_map_reduce`：调 `tensor.select_tree`，得到 `Some t` 就调 `run_levels_*`，得到 `None` 就走 btensor。
- `fuse_producer`：短期保留；以后可以把“融不融”也变成 Mim 里的规则（输入是 producer 和 consumer 的 `OpFacts`）。

---

## 5. 迁移步骤

每一步都可以用优化输出的字节级 oracle（`scripts/oracle.sh`，模式 `opt`）验证“行为不变”，再单独做行为改变的步骤。

| 步骤 | 内容 | 预期 |
| :--- | :--- | :--- |
| P1 | 加 `Rule`、`select_tree`、`OpFacts`；把 `cpu_tree` 和 `generic_tree` 原样包成两条规则，C++ 改为调用 `select_tree` | oracle 字节一致 |
| P2 | 把 `dep`、`unit_b`、`wdim`、`rdim`、`outer`、`red_unit`、`out_unit` 改成 Mim 派生函数；删掉 `PeCpu`/`PeGeneric` 的对应字段和 C++ 计算 | oracle 字节一致 |
| P3 | 把 `generic_tree` 拆成 `rule_transpose`、`rule_row`、`rule_reduce`、`rule_conv_outer`、`rule_leaf`；融合 band 变成 `wrap_fused` | oracle 字节一致 |
| P4 | `sched_valid` 静态校验；`select_tree` 用它过滤；检查条件提成共用小函数 | 字节一致；新增 lit 覆盖“规则拒绝后退到下一条” |
| P5 | `-X tensor:rules=…`、用户规则文件、`tensor.with_sched` | 新增 lit |
| P6 | `conv_impl` 的候选表和选择器；stride-2 卷积自动退回 im2col | 行为改变，配合 `bench/pe_conv` 测量 |

P1–P3 是纯重构，预计 Mim 增加约 100 行、C++ 减少约 100 行；P4 预计 Mim 增加约 80 行。

---

## 6. 需要你决定的问题

1. **规则表的粒度**：规则按“模式”分（收缩、外积、转置、行、归约），还是按“源算子”分（matmul、conv、softmax……）？
   按模式分更通用，也符合 map-reduce 统一表示；按算子分需要在 map-reduce 上保留来源标记。本文按模式分。
2. **选择方式**：只用“第一个合法的规则胜出”，还是给规则加一个静态代价、在多个合法候选里挑最好的？
   先做前者；代价模型可以以后作为 `select_tree` 的另一种驱动加进来。
3. **用户覆盖做到哪一级**：4.5 的三种是否都要，还是先做 `-X tensor:rules`？
