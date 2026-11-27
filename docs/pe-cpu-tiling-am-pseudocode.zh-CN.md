# `run_levels` 抽象机解释器：Python 风格伪代码

本文用 Python 风格的伪代码复述 `src/mim/plug/tensor/tensor.mim` 中 `run_levels_at` 解释器（对应提交 `ad1edc5d2`）。
目的是从抽象机（AM）视角给出一份可以直接对照源码阅读的规格：每条指令对应一个 `exec_*` 函数，`eval` 只做分派。
伪代码省略了类型参数、内存令牌的逐步串联细节和部分 `refly.check` 的报错文本，但保留了全部控制结构和检查条件。

---

## 0. 阅读约定：编译期与运行期

解释器本身是一段普通的 Mim 程序，由部分求值（PE）在编译期执行。
因此伪代码中的值分为两类：

| 记号 | 含义 | 例子 |
| :--- | :--- | :--- |
| 普通 Python 值 | **静态值**：PE 在编译期算出，残余程序里不出现 | 调度树节点、`ts`、`cap`、`dyn`、`part`、`red`、`vdim`、`flat` |
| `Dyn[...]` | **动态值**：运行时才知道，PE 留下计算它的代码 | 运行时形状的 `size`、循环变量 `t`、`lo`（在 `Loop` 内） |
| `emit_*` | 发射残余代码：PE 结束后真正留在 IR 里的循环、分配、分支 | `emit_for`、`emit_if`、`btensor.map_reduce_post` |

只要树是闭合的（`core.pe.is_closed`），`eval` 的 `match` 在编译期完成，抽象机的分派、环境表、续延全部被擦除。
最终残余的只有 `emit_*` 产生的循环嵌套和缓冲区操作。

---

## 1. 指令集与机器状态

### 1.1 指令（`SchedTree`）

```python
class SplitMode(Enum):
    Fixed = 0   # 静态范围；尾块静态剥离
    Loop  = 1   # 运行时数块数；循环体只展开一次
    Peel  = 2   # 静态范围同 Fixed；运行时范围 = 整块循环 + 一个可能为空的边缘块

@dataclass
class Leaf:   vdim: int                                   # 发射微内核；vdim 是向量化的循环（== Rn 表示不向量化）
@dataclass
class Split:  d: int; ts: int; mode: SplitMode; body: Tree # 轴 d 按 ts 切块，串行 affine.For
@dataclass
class Reduce: d: int; w: int; body: Tree                   # 归约轴 d 保留 w 个部分累加器，由 merge 合并
@dataclass
class Pack:   d: int; w: int; slot: int; dep: list[bool]; body: Tree  # 把输入 slot 沿 d 按宽 w 打包成 BLIS 面板
@dataclass
class Stage:  slot: int; body: Tree                        # 物化 producer 在 body 所读盒子上的结果
@dataclass
class Alloc:  slot: int; body: Tree                        # 为 body 里 slot 最外层的 Pack/Stage 一次性分配存储（store_at）
@dataclass
class Switch: d: int; c: int; small: Tree; big: Tree       # 轴 d 的长度 < c 走 small，否则走 big

Tree = Leaf | Split | Reduce | Pack | Stage | Alloc | Switch
```

派生组合子不进入核心 ADT：

```python
def sched_par(sizes, body):        # 多轴 Fixed 切分，第一维在最外层
    for d in reversed(range(Rn)):
        if sizes[d] != 0: body = Split(d, sizes[d], Fixed, body)
    return body
def sched_seq(d, ts, body):  return Split(d, ts, Loop, body)
def sched_peel(d, ts, body): return Split(d, ts, Peel, body)
def sched_levels(levels, vdim):    # IREE 式 lowering_config：每层一个 sched_par
    return foldr(sched_par, levels, Leaf(vdim))
```

树编辑都是 `sched_fold`（自底向上的 catamorphism）的实例：`sched_unpack`、`sched_retile`、`sched_at_leaves`、`sched_with_levels`、`sched_size`。

### 1.2 环境：轴寄存器与输入视图

```python
@dataclass
class Axis:                 # 每个循环轴一个，共 Rn 个
    lo:   Dyn[int]          # 当前子盒在该轴上的起点（相对 origin）
    size: Dyn[int]          # 当前子盒的长度；dyn 为假时是静态值
    cap:  int               # 静态上界，用来给 scratch / panel 定型
    dyn:  bool              # size 是否运行时才知道
    part: bool              # 是否已被某个 Split 切成可能小于全长的块
    red:  int               # Reduce 要求的部分累加器个数，0 表示没有

@dataclass
class View:                 # LevelsView：一个输入 slot 在当前节点之下的样子
    r: int; s: list[int]    # 当前数据缓冲的秩与形状（打包后秩 +1）
    unit: list[int]         # 每轴一步对应多少个元素（打包轴为面板宽 w）
    q: list[AffIndex]       # 视图原点（打包后按面板计数）
    at: Callable            # (q, o) -> 缓冲坐标：从循环坐标到数据坐标的读映射
    data: Buf               # 当前读取的缓冲：原缓冲 / 面板 / Stage 出的盒子
    aff: Aff                # 原始访问映射（数据形式，供盒子计算）
    prod: Prod              # Plain（已有数据）或 Fill(make)（需要 Stage 的 producer）
    reads: list[bool]       # 该 slot 沿哪些循环读取
    flat: bool              # 原缓冲 + 仿射映射：Leaf 可把缓冲偏移到 tile 角点后按线性部分读取
    sn: int                 # 上方 Alloc 提供的存储有多少个元素，0 表示没有
    store: Buf              # 那块一维存储；下一个 Pack/Stage 把它当作自己的面板或盒子
    own: bool               # data 是否仍是 slot 自己的输入缓冲（而不是按 cap 定型的面板或盒子）
```

`own` 决定边缘块能读多远：slot 自己的缓冲在有效长度处结束，面板和物化的盒子按 `cap` 定型（见 §8.3）。

### 1.3 存储与续延

```python
@dataclass
class State:
    axes:  list[Axis]       # 环境 E 之一
    views: list[View]       # 环境 E 之二
    tok:   MemToken         # 存储 S：显式串联的内存令牌
    dst:   Buf              # 存储 S：输出累加缓冲 (Ro, So)

Body = Callable[[State], tuple[MemToken, Buf]]   # 续延 K
```

`eval(tree)` 把一棵树编译成一个 `Body`。
每个 `exec_X(fields..., k)` 接收已经编译好的子树续延 `k`，返回新的 `Body`。

---

## 2. 入口

```python
def run_levels_at(So, Sr, origin,                  # 输出形状、循环范围、循环原点（producer 盒子用）
                  f, init, post,                   # 组合子、单位元、每格后处理
                  merge, lanes, finish,            # 部分结果合并；lanes == 0 表示组合子没有 merge；finish：要跑 epi
                  epi, eager,                      # epilogue（可带外部输入，如 bias）；eager：允许 Leaf 内直接 post
                  drain, drains,                   # 读每个算完的 tile 的消费者（8.5 节）；drains：是否有
                  map_out, maps, prods,            # 输出映射、每个输入的 Aff 读映射、每个输入的 Prod
                  sched, cap0, dyn0,               # 调度树、每轴静态上界、每轴是否运行时
                  tok, dst, xs0):                  # 输出缓冲由调用方给出
    views0 = [View(r=Rs[i], s=Ss[i], unit=[1]*Rn, q=origin,
                   at=levels_plain(maps[i]), data=xs0[i], aff=maps[i], prod=prods[i],
                   reads=aff_reads(maps[i]), flat=aff_linear(maps[i]), sn=0, store=None, own=True)
              for i in range(N)]
    axes0 = [Axis(lo=0, size=Sr[d], cap=cap0[d], dyn=dyn0[d], part=False, red=0) for d in range(Rn)]
    run = lambda tok: eval(sched)(State(axes0, views0, tok, dst))

    red_dyn = any(dyn0[d] for d in range(Ro, Rn))
    if not red_dyn:
        return run(tok)

    # 归约轴有运行时长度：长度为 0 时不运行树，没有 Leaf 写输出，格子直接取 init，再由 epi 映射
    def empty(tok):
        tok = emit_buffer_set(dst, init, tok)
        tok, dst = epi(tok, dst, identity, So) if finish else (tok, dst)
        return (drain(tok, dst, origin, Sr, cap0, dyn0) if drains else tok), dst
    return emit_if(any(Sr[d] == 0 for d in range(Ro, Rn)), then=empty, else_=run)
```

对外的三个包装只是固定部分参数：

| 入口 | 固定的参数 |
| :--- | :--- |
| `run_levels` | 所有 `prods = Plain`，`post = id`，`lanes = 0`，无 epilogue；自己分配输出 |
| `run_levels_into` | `post = id`，`lanes = 0`，`epi = levels_no_epilogue`，`eager = True`；输出缓冲由调用方给出 |
| `run_levels_with` | 分配输出后调用 `run_levels_into` |
| `run_levels_post` | `origin = 0`，其余全部透传，自己分配输出（C++ 的 `lower_to_mem` 用它） |

---

## 3. 分派

```python
def eval(node: Tree) -> Body:            # 仅在 node 闭合时由 PE 展开
    match node:
        case Leaf(vdim):                 return lambda st: exec_leaf(vdim, st)
        case Split(d, ts, mode, body):   return exec_split(d, ts, mode, eval(body))
        case Reduce(d, w, body):         return exec_reduce(d, w, eval(body))
        case Pack(d, w, slot, dep, body):return exec_pack(d, w, slot, dep, eval(body))
        case Stage(slot, body):          return exec_stage(slot, eval(body))
        case Alloc(slot, body):          return exec_alloc(slot, body, eval(body))   # 还要看子树本身，用来算容量
        case Switch(d, c, small, big):   return exec_switch(d, c, eval(small), eval(big))
```

每条指令只修改一部分状态：

| 指令 | 修改 | 不变 |
| :--- | :--- | :--- |
| `Split` | `axes[d]`（lo/size/cap/dyn/part），各 view 的 `q` | `views` 的数据、`dst` |
| `Reduce` | `axes[d].red` | 其余全部 |
| `Pack` | `views[slot]`（换成面板） | `axes`、`dst` |
| `Stage` | `views[slot]`（换成物化盒子） | `axes`、`dst` |
| `Alloc` | `views[slot].sn/store` | 其余全部 |
| `Switch` | `axes[d].cap`（small 分支收紧） | 其余全部 |
| `Leaf` | `dst`、`tok` | `axes`、`views` |

---

## 4. `Split`：切分迭代窗口

```python
def exec_split(d, ts, mode, k):
    check(d < Rn,  "a Split needs an axis below Rn")
    check(ts > 0,  "a Split needs a positive tile size")
    rt_tail, peel = {Fixed: (False, False), Loop: (True, False), Peel: (False, True)}[mode]
    return split_axis(rt_tail, peel, d, ts, k)

def split_axis(rt_tail, peel, d, ts, k):
    def body(st):
        a = st.axes[d]
        n = a.size
        check(not a.dyn or rt_tail or peel, "a Fixed Split needs an axis with a static extent")
        full, tail = n // ts, n % ts                    # a.dyn 时二者是 Dyn
        bound = min(a.cap, ts)
        part  = a.part or a.dyn or ts < n

        def tile(k, t, ext, cap, dyn, tok, dst, xs):    # 运行第 t 块
            views2 = [v._replace(q=shift_q(v, d, t, ts), data=xs[i]) for i, v in enumerate(st.views)]
            # 被打包的轴必须按整面板切：ts % v.unit[d] == 0，否则报 "a Split must cut a packed axis into whole panels"
            axes2 = st.axes.copy()
            axes2[d] = Axis(lo=a.lo + t * ts, size=ext, cap=cap, dyn=dyn, part=part, red=a.red)
            return k(State(axes2, views2, tok, dst))

        def wholes(k, after):                           # 整块：静态无整块时跳过循环
            if not a.dyn and full == 0: return after(st.tok, st.dst)
            return emit_for(full, lambda t, tok, dst: tile(k, t, ts, ts, False, tok, dst, ...), after)

        def tail_static(k):  return lambda tok, dst: tile(k, full, tail, tail, False, tok, dst, ...) if tail else (tok, dst)
        def tail_branch(k):  return lambda tok, dst: emit_if(tail != 0,
                                                             then=lambda: tile(k, full, tail, bound, True, tok, dst, ...),
                                                             else_=lambda: (tok, dst))
        def counted(k):                                 # 运行时数块：块数 = full + (tail != 0)
            return emit_for(full + (tail != 0),
                            lambda t, tok, dst: tile(k, t, min(n - t*ts, ts), bound, True, tok, dst, ...))

        # 运行时范围默认数块，Peel 例外；静态范围只有 Loop 且既有整块又有尾块时才数块
        count = (not peel) if a.dyn else (rt_tail and full != 0 and tail != 0)
        if count:      return counted(k)
        if not a.dyn:  return wholes(k, tail_static(k))   # Fixed / 静态 Peel / 静态 Loop
        return wholes(k, tail_branch(k))                  # 运行时 Peel
    return body
```

要点：
- `Fixed` 在静态范围上把尾块展开成第二份静态代码，所以主块和尾块都是 `dyn = False` 的满速代码。
- `Loop` 只展开一次子树，代价是子树看到的 `size` 是 `Dyn`，必须靠 `cap = bound` 给 scratch 定型。
- `Peel` 在运行时范围上同时得到两者：整块循环是静态 `ts`，只有一个边缘块是 `Dyn`。

---

## 5. `Reduce`：归约拆分

```python
def exec_reduce(d, w, k):
    check(Ro <= d < Rn, "a Reduce needs a reduction loop")
    check(w > 1,        "a Reduce needs at least two partial results")
    def body(st):
        axes2 = st.axes.copy()
        axes2[d] = st.axes[d]._replace(red=w)
        return k(st._replace(axes=axes2))
    return body
```

`Reduce` 只在轴寄存器上记下请求；`red` 沿 `Split` / `Switch` 传给子块，由 `Leaf` 执行（见 §8.4）。
这样做的原因是 `SchedTree (N, Rn)` 的类型固定了循环维数，节点不能给子树新增一条 lane 轴；lane 轴只在 Leaf 发射的局部循环里出现。
merge 是否存在是组合子的性质，不在节点里，检查也放在 Leaf（`lanes == 0` 的实例里如果在这里检查，会在 eval 体内被常量折叠成误报）。

---

## 6. `Pack`：把输入换成 BLIS 面板

```python
def exec_pack(d, w, slot, dep, k):
    def body(st):
        check(d < Rn and w > 0, "a Pack needs an axis below Rn and a positive width")
        v = st.views[slot]
        check(dep == v.reads,              "a Pack's dep must be the loops its slot reads")
        check(v.reads[d],                  "a Pack needs a loop its slot reads")
        check(ready(v.prod),               "a Pack reads a producer slot that no Stage above it computes")
        check(all(st.axes[e].cap < UNBOUNDED for e in range(Rn) if dep[e] or e == d),
              "a Pack sizes its panel by a run-time extent that no Split bounds")

        # 面板形状按静态 cap 定：Rn 维 + 1 条 lane 维；运行时范围多留一个面板给空边缘块
        tok, sp, panel, to = levels_pack_one(v, d, w, dep, sizes(st), caps(st), st.axes[d].dyn, v.sn, v.store, st.tok)
        #   有 Alloc 存储：check(prod(sp) <= v.sn, "a Pack needs more than its Alloc holds")，panel = buffer.view(sp, v.store)
        #   否则：emit alloc panel
        #   然后 fill 整面板；fill 尾面板（lanes = tail）
        v2 = View(r=Rn + 1, s=sp, unit=[w if e == d else 1 for e in range(Rn)], q=[0]*Rn,
                  at=to, data=panel, aff=NONE, prod=Plain, reads=v.reads, flat=False, sn=0, store=None, own=False)
        tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
        return (tok if v.sn else emit_free(panel, tok)), dst
    return body
```

面板坐标：`to(q, o)[k] = o[d] % w`（lane）、`q[d] + o[d] // w`（面板号）、或 `q[k] + o[k]`（`dep[k]` 时）、`0`（不依赖时）。

---

## 7. `Stage`：按需物化 producer 的盒子

```python
def exec_stage(slot, k):
    def body(st):
        v = st.views[slot]
        match v.prod:
            case Plain():                       # 已有数据：什么也不做
                return k(st)
            case Fill(make):
                at = [origin[d] + st.axes[d].lo for d in range(Rn)]
                blo, bext, bcap = aff_box(v.aff, at, sizes(st), caps(st))   # §9
                linear = [aff_span(row, v.aff.vals, at, sizes(st)).affine for row in v.aff.rows]
                # 含 floordiv/ceildiv/mod 的行：盒子宽度取决于 tile 起点，若涉及运行时轴则是 Dyn
                bdyn = [not linear[i] or any(st.axes[d].dyn and mentions(row_i, d) for d in range(Rn))
                        for i, row_i in enumerate(v.aff.rows)]
                if v.sn:                                        # 上方 Alloc 的存储
                    check(prod(bcap) <= v.sn, "a Stage needs more than its Alloc holds")
                    tok, t = st.tok, buffer_view(bext, v.store)
                else:
                    tok, t = emit_alloc(bext, st.tok)
                tok, t = make(blo, bext, bcap, bdyn)(tok, t)    # 只填充；递归：producer 自己也是一次 run_levels_into
                def rel(q, o):                  # 读盒子：仿射行用线性部分，其余行用值减盒子起点
                    x = q + o; y = at + x
                    return [aff_rel(row, x).value if linear[i]
                            else aff_eval(row, v.aff.vals, y) - blo[i]
                            for i, row in enumerate(v.aff.rows)]
                v2 = View(r=v.r, s=bext, unit=[1]*Rn, q=[0]*Rn, at=rel, data=t,
                          aff=v.aff, prod=Plain, reads=v.reads, flat=False, sn=0, store=None, own=False)
                tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
                return (tok if v.sn else emit_free(t, tok)), dst
    return body
```

`make` 来自 `fill_of`（一个 map-reduce producer，用它自己的策略树再调用 `run_levels_at`，`origin = lo`；它的输出映射可以置换输出循环，也可以带一个没有额外输入的 epilogue）、`gather_fill`，或 `pad_fill`（常数 pad：先把这个 box 填成常数，再把 box 与输入重叠的部分拷进来）。
`make` 只负责填充调用方给的缓冲，分配由 `Stage`（或上方的 `Alloc`）负责，这就是 store_at 与 compute_at 的分离。
这就是“从根出发、按需拉取 producer”的融合：producer 只算消费者这块 tile 读到的盒子。

---

## 8. `Leaf`：微内核

### 8.1 检查与读映射

```python
def exec_leaf(vdim, st):
    check(vdim <= Rn, "a Leaf vectorizes a loop below Rn, or none with Rn")
    # 没有 merge 时只能切第一个归约轴，否则会重排组合顺序
    ordered = not any(st.axes[d].part for d in range(Ro + 1, Rn))
    check(lanes > 0 or ordered, "only the first reduction loop may be split when the combiner has no merge")

    axes, vs, tok, dst = st
    lanes_v = axes[vdim].red if vdim < Rn else 0
    check(all(axes[d].red == 0 for d in range(Rn) if d != vdim),
          "a Reduce must split the loop its Leaf vectorizes")

    out = lambda o: map_out([axes[d].lo + o[d] for d in range(Rn)])   # 输出坐标（全局）
    corner = [origin[d] + axes[d].lo for d in range(Rn)]

    # flat：原缓冲 + 仿射映射；把缓冲偏移到 tile 角点映射到的位置，tile 内只用线性部分寻址
    ins_m = [ (lambda o: [aff_rel(row, o).value for row in v.aff.rows]) if v.flat
              else (lambda o: v.at(v.q, o))
              for v in vs ]
    datas = [ buffer.shift(v.data, [aff_span(row, v.aff.vals, corner, [1]*Rn).lo for row in v.aff.rows])
              if v.flat else v.data
              for v in vs ]
    check(all(ready(v.prod) for v in vs), "a Leaf reads a producer slot that no Stage above it computes")

    fresh = all(axes[d].lo == 0 for d in range(Ro, Rn))          # 第一块归约：用 init 起步，而不是读 dst
    whole = eager and not any(axes[d].part for d in range(Ro, Rn))  # 覆盖整个归约：可在 Leaf 内直接做 post
```

### 8.2 发射内核

```python
    def kernel(vec, S, c, at, sz, tok):       # vec=False：范围是运行时值，不向量化
        return btensor.map_reduce_post(
            shape=S, iters=sz, sched=mk_sched(vec=Rn if (not vec or vdim < Ro) else vdim),
            comb=f, acc=(S, at, c, fresh, init), post=(post if whole else id),
            out=at, ins=ins_m, bufs=datas)(tok, c)
```

### 8.3 运行时边缘块

```python
    def edge_tile():
        # 输出轴的 size 是 Dyn：在按 cap 定型的 scratch 上计算，再把有效部分拷回。
        # 只有当每个 slot 都已是按 cap 定型的面板或盒子时，内核才跑满 cap；
        # 只要有一个 slot 还读自己的缓冲（它在有效长度处结束），就只跑有效部分，且不向量化。
        owned   = any(v.own for v in vs)
        full_sz = [axes[d].cap if (d < Ro and not owned) else axes[d].size for d in range(Rn)]
        Sc  = [axes[d].cap  for d in range(Ro)]
        ext = [axes[d].size for d in range(Ro)]
        tok, sc = emit_alloc(Sc, tok)
        tok, sc = emit_copy(dst -> sc, ext, offset=lo, tok)
        tok, sc = kernel(not owned, Sc, sc, loops, full_sz, tok)   # scratch 按循环坐标寻址
        tok, dst2 = emit_copy(sc -> dst, ext, offset=lo, tok)
        return emit_free(sc, tok), dst2
```

### 8.4 `Reduce` 请求的 rfactor

```python
    def rfactor():
        w = lanes_v
        n = axes[vdim].size
        full, tail = n // w, n % w
        # 部分累加器缓冲 p：输出 Ro 维 + 1 条 lane 维（宽 w）
        tok, p = emit_alloc(Sp := [*sizes[:Ro], w], tok)
        # 迭代空间 Rn + 1 维：lane 是第 Ro 维（输出维），vdim 变成“组号”，原坐标 = base + 组号*w + lane
        def part(base, width, groups, seed, tok):
            return btensor.map_reduce_post(iters=[*sizes[:Ro], width, ..., groups, ...],
                                           sched=mk_sched(vec=Ro),     # 向量化 lane 维
                                           comb=f, acc=(p, seed, init),
                                           ins=[lambda c: ins_m[i](back(base, c)) for i in range(N)])(tok, p)
        tok = part(0, w, full, seed=True, tok)  if full else part(0, tail, 1, seed=True, tok)
        if full and tail: tok = part(full * w, tail, 1, seed=False, tok)  # 尾部并入前 tail 条 lane
        used = w if full else tail
        # 合并：沿 lane 维用 merge 归约进 dst；whole 时顺带 post
        tok, dst2 = btensor.map_reduce_post(iters=[*sizes[:Ro], used], comb=merge,
                                            acc=(dst, fresh, init), post=(post if whole else id))(tok, p)
        return emit_free(p, tok), dst2
```

### 8.5 选择路径与 epilogue

```python
    red_v = lanes_v > 1
    check(not red_v or lanes > 0, "a Reduce needs a combiner with a merge")
    edge  = blk(vdim) or blk(vdim + 1)                   # 向量轴或其后一轴的 size 是 Dyn
    check(not edge or all(axes[d].cap < UNBOUNDED for d in range(Ro)),
          "a Leaf computes an edge tile of a run-time extent that no Split bounds")
    # 跑满 cap 的边缘块沿运行时输出轴读到 cap：读这条轴的 slot 必须沿它打包，且面板不窄于 cap，
    # 这样读取留在 tile 起始的那个面板里
    owned = any(v.own for v in vs)
    wide  = all(v.unit[d] >= axes[d].cap
                for v in vs for d in range(Ro) if axes[d].dyn and v.reads[d])
    check(not edge or owned or wide,
          "a Leaf computes an edge tile wider than a panel of a slot it reads along a run-time loop")

    if red_v and not blk(vdim): tok, dst = rfactor()
    elif edge:                  tok, dst = edge_tile()
    else:                       tok, dst = kernel(True, So, dst, out, sizes, tok)

    # 结束这些格子归约的 Leaf 趁数据还在缓存里跑 epi（如 + bias、激活）
    mapped = finish and not whole
    if mapped or drains:
        last = all(axes[d].lo + axes[d].size == Sr[d] for d in range(Ro, Rn))   # 静态或 Dyn
        def finished(tok, dst):
            if mapped: tok, dst = epi(tok, dst, cells, ext)
            if drains: tok = drain(tok, dst, corner, ext, cap, dyn)    # 再把这个 tile 交给每个消费者（输出循环的角点和范围，都是 Nat）
            return tok, dst
        tok, dst = emit_if(last, then=lambda: finished(tok, dst), else_=lambda: (tok, dst))
    return tok, dst
```

`drain` 是单个消费者的 `levels_sink`，或多个消费者用 `levels_drain_both` 串起来；默认关闭，`-X tensor:pe-drain=1` 打开。
`levels_sink` 在这个算完的 tile 上运行消费者自己的 `run_levels_at`：`origin` 设为 tile 的角点，树由 `sink_tree` 给出；绑定到上游某个输出循环的消费者循环只跑 tile 在它上面的范围，其余循环跑完整范围。
叶子只在归约循环的全局坐标（`origin + lo`）为 0 时用 `init` 起头，所以切开一个被绑定的归约循环的那些 tile，会接着前面 tile 留下的值继续累加。
三条检查保证写错的 sink 会报错，而不是静默算错：

```python
    check(all(not tied[d] for d in loops if reads(f_aff, d)),
          "a sink reads its tile along a loop that runs over its whole extent")
    blo, _, bcap = aff_box(f_aff, origin, size, bound)
    check(all(blo[e] == corner[dim_loop[e]] and bcap[e] == cap[dim_loop[e]] for e in range(Ro)),
          "a sink reads other cells than the tile it is given")
    check(all(lanes > 0 for d in loops if tied[d] and d > Rc),
          "a sink cuts a reduction loop after the first without a merge")
```

C++ 的 `LowerToMem::sinks_of` 只负责挑消费者、绑定循环；它挑错了不会导致结果错。

---

## 8.6 `Alloc`：存储位置与计算位置分离

```python
def exec_alloc(slot, tree, k):
    def body(st):
        v = st.views[slot]
        box = lambda caps: prod(aff_reach(row, caps) + 1 for row in v.aff.rows)   # Stage 在给定上界下的盒子大小
        n = levels_need(slot, box, fills=not ready(v.prod), caps=caps(st), dyn=dyns(st), t=tree)
        if n >= UNBOUNDED: n = 0           # 运行时范围没有静态上界：退回每个 Pack/Stage 自己分配
        if n == 0: return k(st)
        tok, store = emit_alloc((n,), st.tok)
        v2 = v._replace(sn=n, store=store)
        tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
        return emit_free(store, tok), dst
    return body

def levels_need(want, box, fills, caps, dyn, t):   # 对子树的静态抽象解释：只追踪 cap
    match t:
        case Leaf():                return 0
        case Split(d, ts, _, b):    return levels_need(..., caps with d -> (ts if dyn[d] else min(caps[d], ts)), b)
        case Reduce(_, _, b):       return levels_need(..., caps, b)
        case Pack(d, w, s, dep, b):
            if s != want:           return levels_need(..., caps, b)
            count = ceil(caps[d] / w) + 1   # 总多留一个面板
            return w * count * prod(caps[e] for e in range(Rn) if e != d and dep[e])
        case Stage(s, b):
            if s != want or not fills: return levels_need(..., caps, b)
            return box(caps)
        case Alloc(s, b):           return 0 if s == want else levels_need(..., caps, b)
        case Switch(d, c, sm, bg):  return max(levels_need(..., caps with d -> min(caps[d], c - 1), sm),
                                               levels_need(..., caps, bg))
```

- 容量是对子树做的一次静态抽象解释：只沿路径收紧 `cap`，取所有最外层使用者的最大值；乘法在 `levels_unbounded` 处饱和。
- 只有最外层的 Pack/Stage 用这块存储；它们给子树的视图 `sn = 0`，所以嵌套的同一 slot 使用者各自分配，不会互相覆盖。
- 同一块存储被 Split 的各个 tile 依次复用：每个 tile 的 Pack/Stage 会重新填满它读到的范围。
- 策略层：`cpu_tree` 只为融合的 slot 在根上加 `Alloc`（覆盖 Stage 的盒子），面板保持原样；`generic_tree` 在 band 循环外为每个融合的 slot 加一个。
  面板提到根上实测没有变快（glibc 对同尺寸分配复用得很好），而且在 512³ 时 clang 会在微内核里多溢出一个寄存器（Zen 5），所以默认策略不用；手写树仍可以用。

## 9. `Switch`：多版本分派

```python
def exec_switch(d, c, small, big):
    check(d < Rn, "a Switch needs an axis below Rn")
    check(c > 1,  "a Switch needs a threshold above 1")
    def body(st):
        a  = st.axes[d]
        lt = st with axes[d].cap = min(a.cap, c - 1)     # small 分支可按 c - 1 给 scratch 定型
        if not a.dyn:                                    # 静态：编译期选一棵子树
            return small(lt) if a.size < c else big(st)
        return emit_if(a.size < c, then=lambda: small(lt), else_=lambda: big(st))
    return body
```

例：`generic_tree` 的外积寄存器块用 `Switch` 把非 2 的幂的边缘逐级切成 2 的幂宽度：

```python
def lane_tree(vl, last, leaf):          # T(w) = Split(last, w, Switch(last, w, T(w/2), leaf))
    t = leaf
    for w in [vl // 16, vl // 8, vl // 4, vl // 2, vl]:
        if w > 1: t = Split(last, w, Fixed, Switch(last, w, t, leaf))
    return t
# 28 = 16 + 12 → 12 < 16 → 8 + 4，每块都是 2 的幂，LLVM 能整体向量化
```

---

## 10. 访问映射 `Aff`：数据形式的仿射表达式

```python
class AffK:  Num(c) | Var(k)            # 系数、除数或模数：字面量，或 vals 中的第 k 个符号
class AffE:  Dim(d) | Sym(k) | Lit(c) | Add(x, y) | Sub(x, y) | Mul(x, k) | FDiv(x, k) | CDiv(x, k) | Mod(x, k)
@dataclass
class Aff:   s: int; vals: list[Dyn[int]]; rows: list[AffE]   # 树是静态的，符号值 vals 可以是运行时值

def aff_eval(e, vals, x):   ...  # 生成 affine.* 下标表达式
def aff_rel(e, x):          ...  # 去掉常数项的线性部分 e(lo + x) - e(lo)；遇到 div/mod 返回“非线性”
def aff_span(e, vals, lo, ext) -> (affine: bool, start, width_minus_1):
    match e:
        case Dim(d):     return True, lo[d], ext[d] - 1
        case Add(x, y):  合并区间
        case Mul(x, c):  区间乘 c
        case FDiv(x, c): (l, w) = span(x); return False, l // c, (l + w) // c - l // c
        case CDiv(x, c): 同上取上整
        case Mod(x, c):  r = l % c; 若 r + w >= c 则绕回，返回 [0, c-1]，否则 [r, r+w]
        case Sub(x, y):  (l, w), (k, v) = span(x), span(y); return False, l - (k + v), (l + w - k) - (l - (k + v))
def aff_reach(e, ext):      ...  # 与起点无关的静态宽度上界，用于 cap；运行时的系数或模数没有静态上界
def aff_reads(f):           ...  # 各行提到的循环
def aff_unit(f, d):         ...  # d 只出现在最后一行，系数为 1
def aff_perm(f):            ...  # 每行对应的循环，以及它们是否是前 m 个循环中互不相同的那些
def aff_box(f, lo, ext, cap):    # 每行的 (起点, 宽度, 静态宽度上界)
    return [span.start], [span.width + 1], [aff_reach(row, cap) + 1]
```

自然数系数保证内层表达式单调，因此 floordiv/ceildiv 的端点映射给出精确盒子；mod 只有在区间落入一个周期时不绕回。
减法是唯一不单调的行（flip 读 `(s - 1) - i`）：它的区间从被减数的最小值减去减数的最大值开始，并且不算仿射，所以 tile 按它的值来读，而不是按线性部分。

`tensor.map_reduce_post` 的映射都以 `Aff` 形式携带：`map_out: Aff(Rn, Ro)`、`maps[i]: Aff(Rn, Ris[i])`、`post_maps[j]: Aff(Ro, Rps[j])`。
impl 直接以数据构造映射，没有其他入口：行是静态的树，用到的每个值（字面量或运行时值）都放在 `vals` 里，所以组合和求值总能展开。
分析读取映射之前，`aff_canon` 把字面量值内联成 `Num`/`Lit`，运行时值保留为符号，并去掉因子或除数 1，让静态上界能看到每个字面量。
`tensor.aff_eval_map` 把数据还原成它描述的函数，供循环嵌套的降级和 btensor 操作使用。
规约通过 `tensor.fold_of (op, h)` 声明它的 monoid；`fuse_tensor` 把生产者内联进 `h` 时保留这一形式。

---

## 11. 策略层（不在 TCB 内）

策略只负责构造树，合法性由解释器的检查保证。

```python
def cpu_tree(...):                       # 收缩（GEMM 及其变体）：BLIS 结构
    leaf = Leaf(m_ax)
    t = peel(m_ax, mr, peel(n_ax, nr, leaf))          # 寄存器块
    t = Pack(a, ...)(t); t = split(m_ax, mc, t)        # A 面板 + L2 块
    t = Pack(b, ...)(t) if keep_b else t               # B 面板（按复用和 4K 别名决定）
    t = Split(k_ax, kc, Loop if dyn_k else Fixed, t)
    t = split(n_ax, nc, t)
    t = Stage(slot, t) 按 band 放在不重复计算的位置   # 融合的 producer
    t = Switch(m_ax, 16*mr + 1, build(pack_b=True), build(pack_b=False)) if m_dyn else build(keep_b)
    return Alloc(slot, t) for each fused slot           # 盒子一次分配，放在所有循环外；面板不提

def generic_tree(...):                   # 其余 map-reduce：reduction、softmax、transpose、gather、attention、直接卷积
    if row:     leaf = Leaf(last)                          # 沿最后一个输出轴向量化
    elif split: leaf = Reduce(inner, lanes, Leaf(inner))   # 连续的归约轴 + 有 merge：rfactor
    else:       leaf = Leaf(Rn)
    base = sched_par(cut(256), sched_par(cut(8), leaf)) if wdim != rdim else leaf   # 转置式双轴切块
    if outer and cpu_conv:                                  # 直接卷积的外积寄存器块
        base = Split(rows, mr, Fixed, lane_tree(vl, last, Leaf(rows)))
    staged = 对每个 fuse 的 slot 包一层 Stage
    return Alloc(fused slots..., sched_par(band of axis 0, staged)) if any_fused else base
```

C++ 侧（`lower_to_mem.cpp`）只做三件事：判定能否走 PE（`pe_generic_plan` / `cpu_tree` 的前提）、收集可融合的 producer（`fuse_producer`、`generic_slots`）并给出 `Prod`、拼出 `run_levels_post` 的调用。

---

## 12. 检查清单

每条检查都在对应指令的 `exec_*` 里，PE 展开时求值；条件为静态假时报错，`-X refly:strict=1` 下变成编译错误。

| 指令 | 检查 |
| :--- | :--- |
| `Split` | 轴在 `Rn` 内；`ts > 0`；`Fixed` 需要静态范围；打包轴按整面板切 |
| `Reduce` | 轴是归约轴；`w > 1` |
| `Pack` | 轴在 `Rn` 内且 `w > 0`；`dep` 等于 slot 实际读的轴；slot 读被打包的轴；slot 已就绪；面板涉及的轴都有有限 `cap` |
| `Stage` | （无：Plain 直接通过） |
| `Alloc` | （无：没有静态上界时退回原行为）；使用它的 `Pack`/`Stage` 检查所需元素数不超过 `sn` |
| `Switch` | 轴在 `Rn` 内；`c > 1` |
| `Leaf` | `vdim <= Rn`；无 merge 时只能切第一个归约轴；`Reduce` 的轴必须是 `vdim`；`Reduce` 需要 merge；读的 slot 都已就绪；运行时边缘块的输出轴有有限 `cap`；跑满 `cap` 的边缘块沿运行时输出轴读的 slot 已沿该轴打包且面板不窄于 `cap` |

---

## 13. PE 之后剩下什么

| 抽象机部件 | PE 后 |
| :--- | :--- |
| `match node`、`eval`、`exec_*` 闭包 | 全部消失 |
| `Axis` 的 `cap` / `dyn` / `part` / `red`、`View` 的 `unit` / `flat` / `prod` / `own` | 全部消失，只影响选了哪条路径 |
| `Axis.lo` / `size` | 静态时折叠为常数，`Loop` 内成为循环变量的仿射式 |
| `View.at`、`aff_eval` | 内联成 `affine.*` 下标表达式，再由 `affine.lower_index` 降成 core 算术 |
| `View.sn` / `store` | `Alloc` 的一次分配；`buffer.view` 降为指针 bitcast |
| `emit_for` / `emit_if` / `emit_alloc` / `btensor.map_reduce_post` | 残余程序：循环嵌套、面板与 scratch 缓冲、微内核 |
