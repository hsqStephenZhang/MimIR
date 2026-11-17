# GPU 多级 tiling 的设计选择：从第一性原理整理

逐点整理，每节一个问题。
用词约定：**tile** 是一段循环范围；**partition** 是把循环范围不重叠地切成 tiles 的整个划分；**用到的子矩阵**是一个 tile 在某个输入数组上读到的区域；**面板**专指搬进 shared memory 的那块子矩阵。

## 1. 一个 level 做什么

出发点是 op 的表示：一段循环范围，每个输入一个 affine map（循环变量 → 数组下标），一个输出 map，一个逐点计算的 body，以及归约运算 `⊕` 和它的种子。
以 GEMM 为例，`C[i,j] = Σ_k A[i,k]·B[k,j]`：循环变量是 `(i, j, k)`，`A`、`B`、`C` 的 map 分别是 `(i,k)`、`(k,j)`、`(i,j)`，body 是乘法，`⊕` 是加法。

每个 level 都是"切分、取子矩阵、可选地搬运、递归、处理结果"，下一层面对的仍是同一形式的 op。

### 1.1 伪代码

```
// Op = { range, in_maps[], out_map, body, ⊕, seed }

run(level, op, ins[], acc) -> out:
    if level 是叶子:
        return leaf(op, ins, acc)

    // 1. 切分：只切这个 level 点名的循环变量，其余原样留给下一层
    tiles = partition(op.range, level.sizes)

    // 2. 输出的 promote（可选）：寿命是整个循环，所以在循环外
    if level.promote_out:
        acc = alloc(level.space, ...)
        acc <- seed 或 load(out)

    // 3. 循环：执行者和顺序由 level 决定
    for tile in tiles  [by level.unit] [order: level.kind]:

        // 3a. slice：不搬数据，只换视图和下标
        sub = restrict(op, tile)
        for a in inputs:
            region[a]      = image(op.in_maps[a], tile)
            view[a]        = slice(ins[a], region[a])
            sub.in_maps[a] = op.in_maps[a] - region[a].origin

        // 3b. 输入的 promote（可选）：真的搬数据，寿命是这一次迭代
        for a in level.promote_in:
            buf = alloc(level.space, region[a].shape, layout)
            copy(buf <- view[a])
            sync
            view[a] = buf

        // 3c. compute：递归到下一层
        part = run(level.next, sub, view, acc 或 acc 的子块)

        // 3d. 这一份结果怎么处理
        acc = combine(level.kind, acc, part, tile)

    // 4. 输出的写回：与第 2 步配对
    if level.promote_out:
        store(out <- acc)
    return acc
```

### 1.2 `partition(range, sizes)`

- **输入**：`range` 给每个循环变量一个区间 `[lo, hi)`；`sizes` 给每个循环变量一个 tile 大小，不切的变量大小取整个区间。
- **输出**：一组 tiles，每个 tile 和 `range` 是同一种东西（每个变量一个区间），所以可以原样交给下一层再切。
- **性质**：tiles 互不重叠，合起来恰好是 `range`。
  这是"每个循环点恰好被算一次"的来源，也是下面三种 `combine` 成立的前提。
- **顺序**：tiles 按下标的字典序排列。
  只有有序归约依赖这个顺序。
- **尾部**：区间长度不被 tile 大小整除时，最后一个 tile 较短。
  怎么处理（缩短、补齐后丢弃多余部分、加判断）是 level 的一个独立选项，不改变上面的性质。

例：`i ∈ [0, 1024)`、`j ∈ [0, 1024)`、`k ∈ [0, 1024)`，`sizes = (128, 128, 不切)`，得到 64 个 tile，每个是 `i` 的 128 行、`j` 的 128 列、`k` 的全部。

### 1.3 `image(in_map, tile)`

- **输入**：一个输入数组的 affine map，和一个 tile。
- **输出**：这个 tile 在该数组上用到的子矩阵，表示为起点和形状。
- **算法**：数组的每一维下标都是循环变量的仿射式，在 tile 上的最小值和最大值在各区间的端点取到（按系数的正负选端点），所以逐维取最小、最大即可，全部在编译期完成。
- **精确性**：
  - map 只是挑选循环变量时（GEMM 的 `(i,k)`），结果恰好是用到的元素，没有多余。
  - map 带偏移时（卷积的 `i + di`），结果是相邻 tile 之间有重叠的矩形。
  - map 带步长时（`2i + di`），结果是包住所有用到元素的最小矩形，里面可能有用不到的元素；这只会多搬，不会算错。
- **不是 partition**：不同 tile 的子矩阵可以重叠。
  GEMM 里同一行范围、不同 `j` 的 tiles 用的是 `A` 的同一块。
  输入是多个单元共同读的，输出才是各单元独占写的。

例：tile 为 `i ∈ [128, 256)`、`j ∈ [0, 128)`、`k ∈ [16, 32)`，则 `A` 的子矩阵是行 128–255、列 16–31（128×16），`B` 的是行 16–31、列 0–127（16×128）。

配套的两个操作：

- `slice(a, region)`：数组 `a` 上的一个视图，不拷贝。
- `in_map - region.origin`：把 map 平移成相对子矩阵起点的下标，这样下一层不需要知道自己在整个数组里的位置。

### 1.4 `combine(kind, acc, part, tile)`

`kind` 同时决定循环的顺序和每个 tile 的结果怎么并入总结果。
它只有三种，分别对应 `tensor.mim` 里 `mod law` 的三条切分引理。

| kind | 切的是哪类变量 | 循环 | `combine` |
|---|---|---|---|
| 并行 | 输出下标里出现的（GEMM 的 `i`、`j`） | 各 tile 并行 | 不做事：每个 tile 已经把结果写进输出里属于自己的那一块 |
| 有序归约 | 被归约掉的（`k`） | 按顺序 | `acc = part`：下一个 tile 在上一个的结果上继续累加，所以 `acc` 是串着传的 |
| 乱序归约 | 被归约掉的（`k`） | 各 tile 并行，各自从种子开始 | 循环结束后用 `⊕` 把各份部分结果合并（树形，或 warp 内的 shuffle） |

各自的前提：

- **并行**：切的变量出现在输出下标里，所以输出也随之被 partition，各 tile 写的位置不重叠。
- **有序归约**：只需要 `⊕` 满足结合律；浮点加法按原顺序累加，结果与不切分时逐位相同。
- **乱序归约**：需要结合律和交换律，种子必须是 `⊕` 的单位元（否则会被并入多次）；浮点结果与原顺序不逐位相同。

### 1.5 `leaf(op, ins, acc)`

- **含义**：不再切分，直接把 `op.range` 里的每个点算掉：按 `in_maps` 从 `ins` 读，执行 body，用 `⊕` 并入 `acc` 在 `out_map` 指定的位置。
- **`acc` 是传入的**：叶子在已有的值上累加，不自己决定种子和写回，这两件事属于上面某一层的第 2 步和第 4 步。
- **三种形态**：
  - 标量循环，累加器在寄存器里。
  - 向量指令：沿存储的最后一维一次读写多个元素，要求 `ins` 的布局让这些元素连续且对齐。
  - 张量核指令（mma、wgmma）：body 与 `⊕` 恰好是"乘后加"，范围恰好是指令的形状时，整段循环换成一条指令；这时 `ins` 和 `acc` 必须已经是指令要求的片段布局，由一个 warp（或 warpgroup）共同执行。
- **叶子的大小由上面各层的 `sizes` 决定**：叶子看到的 `op.range` 就是最后一次切分留下的 tile。

### 1.6 其余操作

| 操作 | 含义 |
|---|---|
| `restrict(op, tile)` | 同一个 body、`⊕` 和 map，循环范围换成 `tile` |
| `[by level.unit]` | 这组 tiles 分给哪一级执行单元（block、warp、thread），或由当前单元自己串行执行 |
| `alloc(space, shape, layout)` | 在 `space`（shared memory 或寄存器）里分配一块，`layout` 决定逻辑下标到存储位置的对应 |
| `copy(buf <- view)` | 把子矩阵搬进 `buf`；它自己是一个恒等的 op，由下一级的执行单元协作完成，也可以再被切分 |
| `sync` | 所有参与填充的单元完成之后，`buf` 才对读者可见，此后只读 |

### 1.7 代入 GEMM

| level | 切 | kind | unit | promote |
|---|---|---|---|---|
| 1 | `i`、`j` → 128×128 | 并行 | block | 无 |
| 2 | `k` → 每步 16 | 有序归约 | block 自己串行 | 输入：`A`、`B` 的面板进 shared memory |
| 3 | `i`、`j` → 8×8 | 并行 | thread | 输出：累加器进寄存器 |
| 叶子 | — | — | — | 8×8×16 的标量或向量循环 |

### 1.8 伪代码没有表达的两处

- **寿命不总是"本 level 的循环"**。
  上表里 level 3 的寄存器累加器属于 thread，却要跨 level 2 的所有 K 步存活。
  所以第 2 步和第 4 步的位置由一个单独的寿命标注决定，而不是固定在本 level 的循环外。
- **时间上不总是嵌套的**。
  流水环把第 3b 步的拷贝提前到前几次迭代发起，第 `i + d` 步的搬运和第 `i` 步的计算重叠。
  数据流仍是这份伪代码，执行顺序不是。

### 1.9 与实现的对应

| 伪代码 | 现在的实现 |
|---|---|
| `run`、三种 `kind` | `mod law` 的解释器与 `split_par`、`split_bound` |
| `partition` | level 链（`TileLevels` 等）里每层的 `sizes` 与 `tail` |
| `image`、`slice` | `tensor.slice` 的范围计算 |
| 输入的 promote | `tensor.stage`，同步协议是 `gpu.superstep` |
| 乱序归约的合并 | `gpu.merge_rounds`、`gpu.lane_merge` |
| `[by level.unit]` | `bind` 标注 |
| `leaf` | 解释器里的 `leaf`，落到 `map_reduce_post` 或张量核指令（`mma`、`wgmma`） |

### 1.10 文献

- 递归分解成同类的更小问题、数据逐层下移：Fireiron（§3）、Sequoia、Goto 与 van de Geijn、BLIS。
- 分块形状由 affine map 决定：Christ、Demmel 等的 HBL 下界；MLIR Linalg 用 tile 在 indexing map 下的像确定 packed tensor 的大小。
- 切分与搬运是两个独立的决定：Halide（计算粒度与存储粒度）、Lift（`toLocal`/`toGlobal` 的语义是恒等）、Exo（`stage_mem`）。
- partition 作为数据侧的收窄、结束处同步：Prism（§3.4.2）。

PDF 在 `docs/papers/`（未提交）。
