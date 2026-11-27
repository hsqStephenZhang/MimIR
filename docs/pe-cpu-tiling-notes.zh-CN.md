# `pe-cpu-tiling` 设计与实现笔记

本文整理 `feat/pe-fuse`（`MimIR-pe-cpu`）中基于部分求值（Partial Evaluation, PE）的一阶调度树 `SchedTree` 与解释器 `run_levels` 的核心直觉、语义、融合机制与 Lean 4 形式化验证潜力。
核心源码位于 `src/mim/plug/tensor/tensor.mim` 与 `src/mim/plug/tensor/phase/lower_to_mem.cpp`。

---

## 1. PE 视角：第一 Futamura 投射

从部分求值的角度，`pe-cpu-tiling` 本质上是第一 Futamura 投射的直接应用：

$$\text{PE}\big(\texttt{run\_levels},\ (\texttt{op\_static},\ \texttt{sched\_tree})\big) \;\longrightarrow\; \lambda\,\texttt{runtime\_args}.\;\text{Loop IR}$$

- **解释器而非手写代码生成器**：`run_levels`（底层实现 `run_levels_at`）是一个普通的 Mim 函数，递归解释一棵一阶代数数据类型 `SchedTree`。
- **静态输入（编译期已知）**：`SchedTree` 本身、算子静态结构（轴数、静态 tile 大小、一阶仿射索引 `Aff`、combiner/epilogue 等）。
- **动态输入（残余参数）**：运行时 buffer 指针、内存令牌 `mem.M 0`、以及动态维度长度。
- **PE 效果**：树遍历、`match` 分派、静态整除/余数拆分、producer 盒推导与索引变换在编译期被完全擦除，残余下来的只有干净的 `affine.For`、`buffer.alloc/free` 与叶子 `btensor.map_reduce_post`。

---

## 2. `SchedTree` 的 7 个节点及伪代码语义

`SchedTree (N, Rn)`（`tensor.mim:1844-1851`）共包含 7 种构造子。
解释器 `run_levels_at`（`tensor.mim:1957-2192`）在递归遍历时维护状态 `State = (axes, views, tok, dst)`：
- `axes[d] = (lo, size, cap, dyn)`：第 `d` 条轴当前块的起点、长度、静态上界与是否运行时动态；
- `views[slot] = (r, s, unit, q, at, data, aff, prod)`：第 `slot` 个输入的视图与 producer 描述；
- `tok: mem.M 0` 与 `dst: Buf`：内存令牌与输出 buffer。

### 2.1 `Leaf(vdim)` — 叶子微内核
```python
def eval_leaf(vdim, axes, views, dst):
    fresh = all(axes[d].lo == 0 for d in red_axes)
    whole = fresh and all(axes[d].size == Sr[d] for d in red_axes)

    if vdim in red_axes and lanes > 1 and not axes[vdim].dyn:
        # 归约轴向量化 (rfactor)：沿 vdim 拆成 lanes 条部分累加器 p，再按 merge 归并回 dst
        p = alloc_partial(axes, lanes)
        run_vec_kernel(p, views, f, init, vec_dim=lanes_axis)
        reduce_merge(dst, p, merge, init if fresh else dst, post if whole else id)
        free(p)
    elif axes[vdim].dyn or axes[vdim + 1].dyn:
        # 动态边缘 tile：拷到静态大小 cap 的 scratch 跑满宽微内核，再把有效 size 拷回 dst
        sc = alloc(cap_out_shape)
        copy_in(sc, dst, axes.lo, axes.size)
        run_kernel(sc, views, f, init if fresh else sc, post if whole else id, vdim=vdim)
        copy_out(dst, sc, axes.lo, axes.size)
        free(sc)
    else:
        run_kernel(dst, views, axes.lo, axes.size, f, init if fresh else dst, post if whole else id, vdim=vdim)
```

### 2.2 `Par(sizes, body)` — 多轴静态切分（静态剥离尾部）
```python
def eval_par(sizes, body, state):
    for d in reversed(range(Rn)):
        if sizes[d] == 0:
            continue
        ts = sizes[d]
        n = state.axes[d].size            # 要求静态已知 (dyn == False)
        full, tail = n // ts, n % ts
        if full > 0:
            for t in range(full):
                eval(body, slice_axis(state, d, t, ext=ts, cap=ts, dyn=False))
        if tail > 0:
            eval(body, slice_axis(state, d, full, ext=tail, cap=tail, dyn=False))
```

### 2.3 `Seq(d, ts, body)` — 单轴统一循环（尾部运行时计算，body 只展开一次）
```python
def eval_seq(d, ts, body, state):
    n = state.axes[d].size
    steps = ceil_div(n, ts)
    bound = min(state.axes[d].cap, ts)
    for t in range(steps):
        ext = min(ts, n - t * ts)
        eval(body, slice_axis(state, d, t, ext=ext, cap=bound, dyn=True))
```

### 2.4 `Peel(d, ts, body)` — 单轴剥离（整块静态快路径 + 动态尾部分支）
```python
def eval_peel(d, ts, body, state):
    n = state.axes[d].size
    full, tail = n // ts, n % ts
    bound = min(state.axes[d].cap, ts)
    for t in range(full):
        eval(body, slice_axis(state, d, t, ext=ts, cap=ts, dyn=False))
    if tail != 0:
        eval(body, slice_axis(state, d, full, ext=tail, cap=bound, dyn=state.axes[d].dyn))
```

### 2.5 `Pack(d, w, slot, dep, body)` — BLIS 风格面板打包
```python
def eval_pack(d, w, slot, dep, body, state):
    v = state.views[slot]
    p, new_idx_map = pack_panel(v.buf, v.at, v.q, axis=d, width=w, dep=dep,
                                size=state.axes.size, cap=state.axes.cap)
    state2 = replace_view(state, slot, buf=p, at=new_idx_map, q=0, unit_d=w)
    eval(body, state2)
    free(p)
```

### 2.6 `Stage(slot, body)` — 生产者按需物化（Producer-Consumer 融合）
```python
def eval_stage(slot, body, state):
    v = state.views[slot]
    if v.prod is Plain:
        eval(body, state)
        return
    blo, bext = aff_box(v.aff, origin + state.axes.lo, state.axes.size)
    _,   bcap = aff_box(v.aff, origin + state.axes.lo, state.axes.cap)
    t = v.prod.make(blo, bext, bcap, bdyn)
    state2 = replace_view(state, slot, buf=t, at=rel_map(v.aff), q=0, prod=Plain)
    eval(body, state2)
    free(t)
```

### 2.7 `Switch(d, c, small, big)` — 按轴长阈值选调度子树
```python
def eval_switch(d, c, small, big, state):
    a = state.axes[d]
    if a.size < c:
        state_lt = tighten_cap(state, d, min(a.cap, c - 1))
        eval(small, state_lt)
    else:
        eval(big, state)
```

---

## 3. 具体例子：BLIS 5 层 GEMM `SchedTree`

对于标准 2D 静态 FP32 GEMM `C[M, N] = A[M, K] @ B[K, N]`（`order = 0`、`dyn_k = 1`，轴 `0 = M, 1 = N, 2 = K`，槽位 `0 = A, 1 = B`），`cpu_tree`（`tensor.mim:2256-2318`）构造如下调度树：

```text
Par(sizes=(0, NC, 0))                                   # 外层 N 循环（步长 NC）
 └─ Seq(d=K, ts=KC)                                     # K 循环（默认 dyn_k=1，避免 K 尾部复制整棵子树）
     └─ Pack(d=N, w=NR, slot=B, dep=(False,True,True))  # 打包 B 的 KC×NC 面板（小 M 或 B 可常驻 cache 时省略）
         └─ Par(sizes=(MC, 0, 0))                       # L2 级 M 循环（步长 MC）
             └─ Pack(d=M, w=MR, slot=A, dep=(True,False,True)) # 打包 A 的 MC×KC 面板
                 └─ Par(sizes=(MR, 0, 0))               # 寄存器级 M 循环（步长 MR）
                     └─ Par(sizes=(0, NR, 0))           # 寄存器级 N 循环（步长 NR）
                         └─ Leaf(vdim=M)                # MR × NR × KC 微内核（沿 M 展开，沿 N = M+1 向量化）
```

- 默认参数（`cache = 256 KiB`、`bytes = 4`）：AVX2 为 `MR=6, NR=16, KC=256, MC=126, NC=1024`；AVX-512 为 `MR=8, NR=32, KC=256, MC=128, NC=1024`。
- `Leaf` 传入 `vdim = M (= 0)`，因为 `btensor` 约定 `vdim` 是微内核外层展开轴（`M`），`vdim + 1`（`N`）是 SIMD 向量化轴。
- 动态维度时：外层 `Par` 自动换为 `Seq`，最内两层 `MR/NR` 的 `Par` 自动换为 `Peel`；若仅 `M` 动态且 `B` 超过 cache，外层再包一层 `Switch(d=M, c=16*MR+1, small=不pack B, big=pack B)`。

---

## 4. PE 分阶段机制：`core.pe.is_closed`、部分静态状态与 `Body`

### 4.1 为什么动态 `tail` 不会破坏 `core.pe.is_closed`？
- `eval` 采用两阶段柯里化：`lam eval (node: SchedTree)@(core.pe.is_closed node): Body`，其中 `Body = State → [mem.M 0, D]`。
- `core.pe.is_closed` 只守在静态递归树结构（`SchedTree`、`AffE`）上，从不检查 `State` 里的运行时值。
- `State` 是**部分静态数据结构（Partially Static Data）**：同一条轴 `(lo, size, cap, dyn)` 里，`cap` 与 `dyn` 是编译期常量（继续驱动静态 `select`/`match` 折叠），`lo` 与 `size` 是运行时 SSA 值（原样透传到残余 IR）。

### 4.2 `Peel` 对动态轴如何生成运行时 `if`？
- 在 `tail_branch`（`tensor.mim:2085-2093`）中，`tail = n % ts` 是运行时值，因此 `(skip, some)#(core.ncmp.ne (tail, 0)) m` 无法在 PE 期折叠，自然残余成 CPS 中的条件分支（`br i1 %cond, %some, %skip`）。
- 整块循环体以 `(ext=ts, cap=ts, dyn=ff)` 调用一次 `k: Body`，特化为无掩码满速静态微内核；`some` 分支以 `(ext=tail, cap=bound, dyn=tt)` 调用第二次 `k`，特化为按静态 `cap` 分配 scratch 的 `edged` 边缘路径。

### 4.3 为什么 `Body` 最终会变成纯 Loop IR？
- `eval sched` 先把 `SchedTree` 递归全部展开成嵌套的普通 `lam (st: State) = ...`。
- `run_levels_at` 最后一行立刻对它应用初始状态 `eval sched (init_axes, vs0, m, dst)`，触发普通 `lam` 的一路 beta 规约与元组投影折叠。
- 规约完毕后，`eval`、`SchedTree`、`Body` 闭包与 `State` 元组全部消失，只留下 `affine.For`、`buffer.alloc/free` 和叶子 `btensor.map_reduce_post`，最终返回 `[mem.M 0, D]`。

---

## 5. 与 AnyDSL / Stincilla HOF 调度库的对比

- **同源处**：底层都依赖 PE 在编译期把调度抽象以零运行时开销擦除。
- **核心区别（Defunctionalization）**：
  - AnyDSL / Stincilla（如 `halide/schedule.impala`、`matmul.impala`）面向库作者，针对具体算子手写并手动组合高阶函数（HOF）交给 PE；
  - `pe-cpu-tiling` 将这些 HOF 再抽象为一阶 ADT（`SchedTree` 与 `Aff`），原来的 HOF（`step_par`、`step_pack`、`step_stage` 等）退居幕后成为统一解释器 `run_levels` 对 ADT 的解释。
- **直接收益**：调度变成了编译器可自动生成（`cpu_tree` / `generic_tree`）、可遍历查询（`sched_splits`、`sched_size`）、可代数改写（`sched_unpack`、`sched_retile`、`sched_at_leaves`）的一阶数据，并把循环切分、BLIS 面板重排与跨算子融合统一在同一套状态机里。

---

## 6. `Pack` 临时 Buffer 的 Hoist 现状

- **当前不支持显式 hoist**：在 `SchedTree` 层，`Pack` 把 `buffer.alloc`、`fill`、`body` 与 `buffer.free` 绑定在同一个节点里，没有类似 Halide `store_at` / `compute_at` 的分离原语。
- **后端优化现状**：`mem` 插件里的 `Heap2Stack`（`src/mim/plug/mem/phase/heap2stack.cpp`）只提升循环外（`!in_loop`）且元素数 `<= 1024` 的分配；`panel_b`（1 MiB）和 `panel_a`（128 KiB）既在循环内又超阈值，当前落地为循环内 `malloc`/`free`。
- **演进前提已就绪**：`levels_pack_one`（`tensor.mim:1819`）已经用静态上界 `cap` 计算面板形状 `sp`，使 buffer 大小本身成为循环不变量，后续无论是拆分 `Alloc` 节点还是在 `mem` 层增加静态 buffer hoisting pass 都很顺畅。

---

## 7. `Stage` 融合的 5 项基础设施

1. **一阶访存表达式 `Aff`（`tensor.mim:1666-1765`）**：由 impl 直接以数据构造，提供 `aff_box`（无 `floordiv`/`mod` 时做精确盒推导，求出上游 `(blo, bext, bcap, bdyn)`）与 `aff_rel`（去掉起点常量偏移，供下游以 0 为原点读局部 buffer）。
2. **按需生产者抽象 `Prod = Plain | Fill`（`tensor.mim:1768-1791, 2349-2366`）**：`Fill make` 接收任意子盒 `(lo, ext, cap, dyn)` 并在内部递归调用 `run_levels_at(..., origin = lo)` 与 `tree_of(box_size, box_dyn)`，从而天然支持多级嵌套融合与 `gather_fill`。
3. **`step_stage` 与 `refly.check` 静态防呆（`tensor.mim:2124-2149`）**：`Stage` 物化局部临时 buffer 后将 `views[slot]` 切换为相对视图并把 `prod` 置为 `Plain`；`Leaf` 与 `Pack` 入口通过 `refly.check(levels_ready (v#prod))` 在编译期拦截漏插 `Stage` 的非法调度树。
4. **`LowerToMem` 的 producer 识别与 `anchors_`（`lower_to_mem.cpp:515-642`）**：筛选 `aff_boxable`、`map_out == id`、归约轴静态、无外部 epilogue 的上游（contraction 限单消费者，普通 map-reduce 允许最多 2 消费者重算），用占位 buffer 跳过全量物化，并通过 `anchors_` 将上游普通外部输入锚定到最外层消费者入口，防止 `add_mem` 在每个 tile 中重复计算。
5. **Band 级 `Stage` 放置策略（`tensor.mim:2307-2315, 2333-2341`）**：`cpu_tree` 把 `Stage(slot)` 放在该槽位依赖的并行轴（A 按 `MC`、B 按 `NC`）之下、所有无关循环之上，确保每个上游块只算一次；若上游 contraction 单条 band 达到 `2 * cache` 以上，则自动退回树根一次性 `Stage`。

---

## 8. Lean 4 形式化验证潜力（对照 `cgo27-paper248.pdf`）

### 8.1 直接承接论文 P4 的验证分解
`cgo27-paper248.pdf`（Section 3, P4, p.4）指出，MimIR 的 PE 范式把编译器验证分解为三项义务：
1. PE 正确性（全局一次性证明）；
2. 函数调用替换正确性（全局一次性证明）；
3. 目标函数与参考函数的等价性（如 `loop ≡ tile`）。

套用到 `pe-cpu-tiling` 上，第 3 项进一步升级为对所有良构调度树的全称定理：

$$\forall\, t : \texttt{SchedTree},\quad \texttt{WF}(t) \;\Longrightarrow\; \llbracket \texttt{run\_levels}(t, \text{op}) \rrbracket \;=\; \llbracket \texttt{map\_reduce}(\text{op}) \rrbracket$$

由于 `SchedTree` 和 `Aff` 均为一阶归纳 ADT，该定理在 Lean 4 中直接转化为**对 `t : SchedTree` 的 7 节点结构归纳**，完全避开 CFG、SSA dominance 与 region 语义的复杂性。

### 8.2 7 个节点的模块化证明义务
以归纳不变式“在当前子盒 `[lo, lo + size)` 上恰好把 `map_reduce` 的贡献累加进 `dst`”为核心：
- **`Par` / `Seq` / `Peel`**：区间划分引理（互不相交且覆盖 `[lo, lo + n)`；并行轴无额外前提，归约轴只需 combiner 结合律）。
- **`Switch`**：双分支归纳假设 + 在 `size < c` 下保持 `size <= min(cap, c - 1)`。
- **`Pack`**：读写索引互逆引理 `panel[to(q, o)] = src_buf[ix(q, o)]` 与 `cap` 边界安全性。
- **`Stage`**：对一阶 `AffE` 做结构归纳，证明盒推导健全性（可用 Lean 4 `omega` 自动消解大部分算术目标）：
  $$\forall\, x \in [\text{lo}, \text{lo}+\text{size}),\quad \texttt{aff\_eval}(f, x) \in \texttt{aff\_box}(f, \text{lo}, \text{size}) \;\land\; \texttt{aff\_eval}(f, x) - \texttt{blo} = \texttt{aff\_rel}(f, x - \text{lo})$$
  由此任意深度的嵌套 `fill_of` 融合正确性随结构归纳自动成立。
- **`Leaf`**：基例（`rfactor` 依赖结合交换幺半群 + `init` 单位元；`edged` 依赖 `size <= cap` 与有效区拷回）。

### 8.3 研究潜力与论文定位
1. **搜索 / Agent 与正确性彻底解耦**：主定理证毕后，整个调度生成与改写侧（`cpu_tree`、`generic_tree`、`sched_retile`、`sched_unpack`、未来的 auto-tuner 或 LLM Agent）都无需纳入可信计算基（TCB）；只要产出树通过极小的可执行 `WF(t)` 检查（与现有 `refly.check` 约束高度重合），即由构造保证正确，直接兑现论文 Intro 提到的 *"automation with confidence, increasingly driven by agentic workflows"*。
2. **回应论文 P1 的 Open Problem**：论文第 3 页指出模块化刻画 dialect 语义并组合进编译器正确性证明仍是开放问题；`SchedTree` + `Aff` + `run_levels` 恰好把多级 tiling、动态尾部、BLIS packing 与跨算子融合统一进一个 7 节点小代数中，并可与 GPU 侧 `proofs/ring/` 的异步环与 barrier 协议证明形成 CPU（数据代数等价）+ GPU（异步同步安全）的互补体系。

### 8.4 需要明确的 3 个真实边界
1. **浮点非结合性**：切归约轴与 `rfactor` 改变累加顺序，形式化等价成立在精确代数/实数（fast-math）语义下，而非逐 bit IEEE-754 等价。
2. **Lean 模型与 Mim 实现的同步**：需通过从 Lean 抽取代码、可执行校验或规范对齐来收窄 Lean 模型与 `tensor.mim` 实现之间的信任边界。
3. **内存隔离与非仿射扩展**：`buffer.alloc/free` 生命周期无别名、`anchors_` 求值锚定及 `gather_fill` 间接索引，在纯函数张量模型之外需要显式不交内存模型或分离逻辑覆盖。
