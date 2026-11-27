# `SchedTree` 与 `run_levels` 的 Lean 4 形式化方案

本文分析如何在 Lean 4 中验证 `pe-cpu-tiling`：`src/mim/plug/tensor/tensor.mim` 中的一阶调度树 `SchedTree`、一阶仿射访问映射 `Aff`，以及在 bufferization 时由部分求值（PE）展开的解释器 `run_levels_at`。
内容包括与 `cgo27-paper248.pdf`（*Principles for Multi-Level Compiler Design*）的衔接、规格与归纳不变式、逐节点证明义务、良构条件、研究价值与可信边界，以及分阶段路线。
文中的代码行号指 `tensor.mim` 当前版本（`cca658689`）。

---

## 1. 与论文验证框架的衔接

### 1.1 P4 的三项证明义务
论文 Section 3 P4（*Minimize Compiler Writing*，p.4）指出，在 MimIR 的 PE 范式下，验证一项变换分解为三项义务：

1. **PE 正确性**：部分求值器实现程序语义。
2. **rewriter 正确性**：把一个函数调用替换为另一个保持语义。
3. **函数等价性**：论文的例子是 `loop` ≡ `tile`。

论文把 (1)(2) 当作可被所有基于 rewriting 与 PE 的优化复用的论证提出，并未声称已经完成。
P1（p.3）只说明 MimIR **核心**的语义与类型系统已形式化并证明了类型安全，dialect 语义的模块化规格仍是开放问题。
因此本文的全部结论都以 (1)(2) 为前提，只处理第 (3) 项。

### 1.2 与论文第 7 节 `Tileable` 的区别
论文第 7 节的调度是 `Schedule = Tileable → Tileable` 的项，共五个（两种 strip-mine、interchange、选 `vdim` 与 `unroll` 的两个）。
每个调度可以单独证明“对所有 `t` 保持语义”，组合随之成立，所以论文路线本身也给出全称结论。
`SchedTree` 的区别在于证明对象的位置：

- 论文路线：每个变换各证一次，再证 `retile` 与 nest builder。
- `SchedTree` 路线：调度被去函数化为一阶数据 `SchedTree`（7 个构造子）与 `AffE`（8 个构造子），所有语义集中在一个解释器 `run_levels_at` 里，只证它一次。

代价是这个解释器远比论文的 strip-mine 复杂。
论文的 strip-mine 要求 tile 大小整除轴长（`tensor.mim:758` 的 `refly.check`），而 `run_levels` 处理运行时长度、尾块、BLIS 打包、producer 融合、`rfactor` 与 epilogue；这些既是表达力的来源，也是证明的主要难点。

另一处差异与 P4 的方向相反。
论文强调 `Tileable` 的一致性条件写在依赖类型里，rank 不一致的实现无法通过类型检查。
`SchedTree` 的轴号却是 `d: Nat`，由 `refly.check (d < Rn)` 检查（`Seq`/`Peel`/`Pack`/`Switch`，`tensor.mim:2117`、`2176`–`2186`）。
把轴号改为 `Idx Rn`，这类条件便由类型保证，`WF` 只剩下真正的语义前提（见第 5 节）。

---

## 2. 规格与 Lean 建模

### 2.1 一阶语法对象
```lean
inductive AffE (n s : Nat) where
  | dim  : Fin n → AffE n s
  | sym  : Fin s → AffE n s
  | lit  : Nat → AffE n s
  | add  : AffE n s → AffE n s → AffE n s
  | mul  : AffE n s → Nat → AffE n s
  | fdiv : AffE n s → Nat → AffE n s
  | cdiv : AffE n s → Nat → AffE n s
  | mod  : AffE n s → Nat → AffE n s

inductive SchedTree (N Rn : Nat) where
  | leaf   : (vdim : Nat) → SchedTree N Rn
  | par    : (sizes : Fin Rn → Nat) → SchedTree N Rn → SchedTree N Rn
  | pack   : (d : Fin Rn) → (w : Nat) → (slot : Fin N) → (dep : Fin Rn → Bool) → SchedTree N Rn → SchedTree N Rn
  | seq    : (d : Fin Rn) → (ts : Nat) → SchedTree N Rn → SchedTree N Rn
  | peel   : (d : Fin Rn) → (ts : Nat) → SchedTree N Rn → SchedTree N Rn
  | stage  : (slot : Fin N) → SchedTree N Rn → SchedTree N Rn
  | switch : (d : Fin Rn) → (c : Nat) → (small big : SchedTree N Rn) → SchedTree N Rn
```
Lean 侧的轴号直接取 `Fin Rn`，对应第 1.2 节建议的 `Idx Rn`；与 Mim 源码的这一差异需在模型对齐时显式说明。

### 2.2 规格：`map_reduce_post`
`run_levels_at` 的参数比纯 `map_reduce` 多出 `post`、`merge`、`lanes`、`finish`、`map_out`、`origin` 与 `prods`。
规格应是带 epilogue 的折叠：

$$\text{spec}(o) \;=\; \text{post}\Big(\textstyle\text{foldl}_{\,r \,\in\, R\ \text{按字典序}}\ f\big(\cdot,\ \text{ins}(o, r)\big)\ \text{init}\Big)$$

其中 `o` 遍历输出坐标，`R` 是归约域，`ins` 经各槽位的 `Aff` 读取输入；`finish = ff` 时（例如 `fill_of` 中的 producer）`post` 不在此处应用。

### 2.3 解释器状态
- `axes : Fin Rn → Axis`，每条轴有 `lo, size, cap : Nat` 与 `dyn, part : Bool`，满足 `size ≤ cap`；
- `views : Fin N → View`，每个槽位或已就绪（`Plain`，持有缓冲区与读取映射 `at`），或是尚待 `Stage` 的 producer（`Fill make`）；
- `dst`：输出缓冲区。

`Fill make` 不在 Lean 中展开，而是作为**语义前提**：

> `make(lo, ext, cap, dyn)` 在 `ext ≤ cap` 时返回的缓冲区 `B` 满足 `B[y] = P[lo + y]`，其中 `P` 是该 producer 的规格结果。

`fill_of` 与 `gather_fill` 都按这条前提接入，`gather_fill` 是纯函数，无需扩展内存模型。

### 2.4 归纳不变式：三状态输出单元
`dst` 中每个输出单元处于三种状态之一：

1. **未初始化**：输出缓冲区通常不先写入 `init`（只有存在运行时归约长度时才写，`tensor.mim:2192`）。
2. **已累加前缀**：单元值等于对归约域某个字典序前缀做 `f` 折叠的结果。
3. **已完成**：单元值等于 `spec(o)`，即已应用 `post`。

叶子的行为决定状态转移：

- 所有归约轴 `lo = 0` 的叶子（`fresh`）忽略单元原值，从 `init` 起累加（状态 1 或 2 → 2）；
- 其余叶子在原值上继续累加（状态 2 → 2）；
- 覆盖整个归约的叶子（静态的 `whole`）或在 `finish` 下结束归约的叶子（`last`：所有归约轴 `lo + size = Sr`）随后应用 `post`（→ 3）。

不变式陈述为：

> 在 `WF(t)` 下执行 `eval t`，对当前子盒 `∏_d [lo_d, lo_d + size_d)` 覆盖的每个输出单元，执行后的状态等于把该子盒的归约贡献接在执行前状态之后的结果；子盒之外的单元不变。

这条不变式依赖一个**跨节点的顺序引理**：

> 对每个输出单元，覆盖它的叶子按执行顺序恰好把归约域划分为字典序递增的若干段，第一段的所有归约 `lo` 为 0，最后一段的所有归约轴到达 `Sr`。

所有循环都是串行升序，因此该引理成立，但它不是某个节点的局部性质，必须单独证明，并在 `Par`/`Seq`/`Peel`/`Switch` 的情形中被引用。

---

## 3. 逐节点证明义务

| 节点 | 证明义务 | 前提 |
| :--- | :--- | :--- |
| **`Par` / `Seq` / `Peel`** | 区间划分：`[lo, lo + n)` 切成按序排列、互不相交、并集覆盖全区的子区间；`Peel` 的整块循环 + `tail ≠ 0` 尾块与 `Seq` 的计数循环共享同一划分；`part` 标志与静态 `whole` 一致。 | `ts > 0`。单根归约轴按序切分只是把左折叠拆成连续几段，**不需要结合律**。多根归约轴的分块交换了访问顺序（例如 conv 的 `kh`/`kw`/`ci`），需要累加可交换：`f(f(a, x), y) = f(f(a, y), x)`。 |
| **`Switch`** | 分支等价与上界收紧：`size < c` 时把 `cap` 收紧为 `min(cap, c - 1)` 仍满足 `size ≤ cap`，两分支由归纳假设得证。 | `c > 1`。 |
| **`Pack`** | 读写互逆：`levels_pack_one` 写出面板 `p` 后，下游按 `to(q, o)` 读到 `p[to(q, o)] = x[ix(q, o)]`；`to` 落在以静态 `cap` 定界、运行时长度另加一块备用面板的范围内；替换 `views[slot]` 后视图不变式保持。 | `w > 0`；槽位已就绪；内层切分步长是 `w` 的倍数（`split_serial` 检查 `ts % unit[d] = 0`）；**`dep` 覆盖访问映射的依赖**：`dep[j] = ff` 时面板在轴 `j` 上长度为 1、下标恒为 0，所以槽位的 `Aff` 不得依赖轴 `j`。 |
| **`Stage`** | 盒推导健全性（`aff_box_sound`，见下）加上 `Fill` 语义前提，得出物化后的新视图满足视图不变式；读取映射换成 `aff_rel` 给出的线性部分。 | 访问映射可盒化（无 `FDiv`/`CDiv`/`Mod`）。嵌套 `fill_of` 的内层树由 `tree_of` 新生成，**不是 `t` 的子树**，由第 3.2 节的外层归纳处理。 |
| **`Leaf`** | 普通路径：`map_reduce_post` 在子盒上实现第 2.4 节的状态转移。`edged` 路径：scratch tile 按 `cap` 分配，拷入拷出只覆盖 `[0, size)`。`rfactor` 路径：按 `lanes` 交错拆分归约轴，各 lane 从 `init` 起累加，最后用 `merge` 并入单元，等价于顺序折叠。 | 槽位全部就绪。`edged` 在输出轴上按 `cap` 计算（`full_sz`，`tensor.mim:1991`），越出 `size` 的部分只读、结果丢弃，所以要求这些读取落在缓冲区内（`cpu_tree` 通过在运行时长度下打包来保证）。`rfactor` 要求 `(To, merge, init)` 是交换幺半群且 `f(a, x) = merge(a, f(init, x))`。 |

### 3.1 关键引理：`aff_box_sound`
```lean
theorem aff_box_sound {n m : Nat} (f : Aff n m) (lo ext : Fin n → Nat)
    (hok : aff_boxable f = true) :
    let (blo, bext) := aff_box f lo ext
    ∀ x : Fin n → Nat, (∀ d, x d < ext d) →
      (∀ i, blo i ≤ aff_eval f (fun d => lo d + x d) i ∧
            aff_eval f (fun d => lo d + x d) i < blo i + bext i) ∧
      (∀ i, aff_eval f (fun d => lo d + x d) i - blo i = aff_rel f x i)
```
在 `Nat` 上所有系数非负，表达式在 `lo` 处取最小值，所以引理成立。
但 `mul` 的系数是变量，`x * c` 超出 `omega` 的线性片段；各情形需要 `Nat.add_mul`、`Nat.mul_le_mul_right` 等单调性与分配律引理，或借助 `nlinarith`。
`ext d = 0`（运行时的空尾块）时，`aff_span` 的饱和减法给出 `bext = 1`，盒子偏大但无害；Lean 的定义要与之一致，引理中的全称量化在此情形下为空真。

### 3.2 融合的外层归纳
主定理以 `Fill` 语义前提为假设（第 2.3 节）。
`fill_of` 的 `make` 本身调用 `run_levels_at`，其 `post = id`、`finish = ff`、归约覆盖完整的 `Sr`，所以对 producer 应用主定理即可消去该假设。
整体是一个嵌套归纳：外层对融合深度（producer 的嵌套层数），内层对 `t` 的结构。

---

## 4. 边界情况：运行时长度为 0 的归约
归约轴为运行时长度时，`dst` 先写入 `init`，以便长度为 0 时输出仍有定义（`tensor.mim:2191`）。
但长度为 0 时不运行任何叶子，`post` 也不会被应用，输出是 `init` 而不是 `post(init)`。
规格若按第 2.2 节定义，`post` 不把 `init` 映射到自身时（例如带 bias 的 epilogue）两者不一致。
这一点需要核对：要么在规格中约定此情形，要么修改代码；形式化正是用来暴露这类情形的。

---

## 5. 良构条件 `WF`：结构检查与语义前提

`WF` 分为两层。

### 5.1 结构检查：已由 `refly.check` 实现

| 约束 | `tensor.mim` 中的检查 |
| :--- | :--- |
| 叶子与 `Pack` 只读已就绪的槽位 | `levels_ready`（`tensor.mim:1977`、`2119`） |
| 打包后的轴只按整块面板切分 | `split_serial` 检查 `ts % unit[d] = 0` |
| `Par` 不切运行时长度的轴 | `split_serial` 中对 `dyn` 的检查 |
| `Stage` 的访问映射可盒化 | `aff_box` 的 `ok`（无 `FDiv`/`CDiv`/`Mod`） |
| 参数范围 | `Seq`/`Peel`：`d < Rn`、`0 < ts`；`Pack`：`d < Rn`、`0 < w`；`Switch`：`d < Rn`、`1 < c` |

轴号改为 `Idx Rn` 后，表中的 `d < Rn` 各项转由类型保证。

### 5.2 语义前提：代码中没有检查

| 前提 | 来源 | 违反时的后果 |
| :--- | :--- | :--- |
| `Pack` 的 `dep` 覆盖槽位 `Aff` 依赖的轴 | C++ 计算的 `plan.dep`（`lower_to_mem.cpp:552`、`663`），`step_pack` 不校验 | 静默地读错数据 |
| `map_out` 在输出轴上单射且不依赖归约坐标 | 调用方 | 不同 tile 写同一单元，`fresh` 与“子盒外不变”失效 |
| `Fill` 语义前提 | producer 的实现 | `Stage` 物化出错误数据 |
| `rfactor`：`(To, merge, init)` 是交换幺半群且 `f(a, x) = merge(a, f(init, x))` | 调用方声明 `lanes > 1` | 拆分归约改变结果 |
| 多根归约轴被重排时，累加可交换 | 调用方 | 同上 |
| `edged` 路径中越出 `size` 的读取落在缓冲区内 | `cpu_tree` 的打包策略 | 越界读取 |

`dep` 一项可以就地补上：在 `step_pack` 中用 `aff_coef` 检查 `dep[j] = ff` 的轴在所有行中系数为 0，把一项 C++ 保证的事实变成 `refly.check`。
其余几项属于代数性质或调用约定，需要作为定理假设，或在构造 `combiner` 时由类型携带。

因此，主定理的准确形式是：

$$\texttt{WF}_\text{struct}(t) \;\wedge\; \texttt{Sem}(\text{op}) \;\Longrightarrow\; \llbracket \texttt{run\_levels\_at}(t, \text{op}) \rrbracket \;=\; \texttt{spec}(\text{op})$$

---

## 6. 研究价值与论文定位

### 6.1 调度策略与正确性解耦
论文 Introduction（p.1）把更强的形式化基础与“increasingly driven by agentic workflows”的自动化联系在一起。
在本方案下，调度策略层（`cpu_tree`、`generic_tree`、`sched_retile`、`sched_unpack`、`sched_at_leaves`，以及未来的 auto-tuner、搜索或 LLM Agent）相对于 `run_levels_at` 的语义移出了可信基：它们生成的任意树只要满足 `WF_struct`，且算子满足 `Sem`，结果就正确。
可信基仍包括 P4 的 (1)(2)，即 MimIR 的 PE、normalizer、`LowerToMem`、`btensor.map_reduce` 的 lowering 与后端，以及 Lean 模型与 Mim 源码的对应关系。

### 6.2 回应 P1 的开放问题
P1（p.3）的开放问题是如何模块化地规定 dialect 的语义并组合进编译器正确性证明。
`SchedTree` + `Aff` + `run_levels_at` 提供了一个具体例子：多级 cache 分块、运行时尾块、BLIS 打包与 producer 融合都由一个一阶数据类型描述，并由一份嵌套归纳证明覆盖。
它与 `MimIR-thoughts/proofs/ring/` 中的 Lean 4 GPU ring 校验器互补：

- CPU `SchedTree`：循环切分、数据重排（`Pack`）与局部物化（`Stage`）的**值等价**；
- GPU ring：多阶段异步搬运（TMA/`cp.async`）与 `mbarrier` 握手的**协议安全**（无竞态、无死锁）。

### 6.3 浮点结论可以更强
单根归约轴的 `Par`/`Seq`/`Peel` 切分保持顺序折叠，对 IEEE-754 也逐位精确，GEMM 沿 `K` 分块就是这种情况。
只有 `rfactor` 与多根归约轴的重排改变运算顺序，需要实数语义或显式的 fast-math 假设。
这比“切分归约轴即改变结合顺序”更精确，也是一个可以在论文中陈述的性质。

---

## 7. 可信边界

1. **浮点**：见第 6.3 节，非逐位精确的只有 `rfactor` 与多归约轴重排。
2. **Lean 模型与 Mim 源码的差距**：Lean 证明的是抽象的 `run_levels_at`。缩小差距的途径包括：在 Lean 中运行可执行的 `WF_struct` 检查器核验编译期导出的树（复用 `show_tree`）、对两个解释器做差分测试，或长期从 Lean 生成 Mim 解释器。
3. **显式内存**：`buffer.alloc`/`buffer.free` 的作用域安全、外部输入锚定（`anchors_`）的求值时机，以及 `edged` 路径读取未初始化的面板内存，超出纯函数张量模型；纳入端到端证明需要不交内存模型或分离逻辑。

---

## 8. 分阶段路线

0. **阶段 0：随机差分测试**。随机生成满足 `WF_struct` 的 `SchedTree`，对照直接执行的 `map_reduce_post` 比较结果，做成 `mim-test` 用例。成本最低，并能检验第 5.2 节的前提，例如故意传错 `dep` 应得到错误结果，补上 `aff_coef` 检查后应在编译期报错。第 4 节的边界情况也应在这里落定。
1. **阶段 1：值语义核心**。形式化 `AffE`、`aff_box`、`aff_rel` 并证明 `aff_box_sound`；在纯张量语义下证明 `Par`/`Seq`/`Peel`/`Switch`/`Stage`/`Leaf`（普通路径）的主定理，包括三状态不变式与顺序引理。预计 1.5k–3k 行 Lean。
2. **阶段 2：`Pack`、`edged` 与 `rfactor`**。加入 `levels_pack_one` 的面板索引引理（含运行时长度的备用面板）、`edged` 的拷入拷出引理与交换幺半群下的 `rfactor` 引理，覆盖全部 7 个节点，并完成融合的外层归纳。
3. **阶段 3：与编译器联通**。由 `LowerToMem` 导出实际构造的 `SchedTree` 与 `Aff`，用 Lean 编译出的 `WF_struct` 检查器核验全部 lit 回归与外部搜索得到的调度。
