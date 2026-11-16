# tiling 编译流水线：伪代码版（2026-11-16）

本文是 [`tile-pipeline.zh-CN.md`](tile-pipeline.zh-CN.md) 的伪代码版本：不贴 IR，而是用类 Python 的伪代码写出每个阶段的输入/输出形态与变换算法的骨架，再把三个示例（SIMT、MMA、wgmma + warp specialization）从源程序逐阶段展开到 PTX。
IR 的真实片段、复现命令见原文；本文的伪代码均对照 `src/mim/plug/tensor/tensor.mim`、`tensor/phase/{tile_fuse,lower_to_mem}.cpp`、`gpu/phase/{outline,erase_phases}.cpp`、`nvptx/phase/nvptx.cpp` 写成，行号为近似值，以函数名为准。

记号：

- `for t in grid(ext, ts):` 遍历 `ext` 按 `ts` 切出的所有 tile，`t.off`/`t.ext` 是 tile 的原点与尺寸；
- `@group`/`@item`/`@warp`/`@warpgroup`/`@specialized` 标注循环绑定到的硬件层级（即 `gpu.bind.*`），`@coop`/`@chained`/`@async` 是 work-group 内所有 work-item 分摊的协作循环；
- `smem[...]` 是共享内存，`barrier()` 是 work-group 屏障（`gpu.fence` → `gpu.sync_work_items` → `bar.sync 0`）；
- `⟦e⟧` 表示 IR 中的值 `e`，`A ⊕ B` 表示按 tile 拼接。

## 1. 整条流水线

```python
def compile(prog):                                   # opt.mim 的 _default_compile，只列与 tiling 相关的阶段
    prog = raise_attention(prog); prog = group_attention(prog)
    prog = optimize(prog)                            # PE：源程序里的 law.run 在这里就被展开成实现原语
    prog = reassoc(prog); prog = lower_tensor(prog); prog = fuse_tensor(prog)
    prog = group_default(prog)                       # tile-fuse=1：TileFuse 兑现每个 tile_req（§3、§4）
    prog = lower_to_mem(prog)                        # 张量原语 → buffer + affine.For + gpu.*（§6）
    check_sync(prog); prog = erase_phases(prog)      # 工作组阶段令牌 → fence / panel_scope（§7.1）
    prog = distribute(prog); prog = add_mem(prog)
    prog = lower_btensor_map_reduce(prog)            # 叶子 btensor.map_reduce_post → 嵌套 affine.For
    mem_checks(prog); prog = buffer_lower_ptr(prog)  # Buf → Ptr，buf_ldmatrix → ldmatrix（§7.1）
    prog = gpu_outline(prog)                         # bind.group 循环 → kernel + launch（§7.2）
    prog = lower_index(prog); prog = cps_conv(prog); prog = lower_for(prog); ...
    return nvptx_emit(prog)                          # split_apply → NVVM IR → opt/llc/ptxas/fatbinary（§7.3）
```

## 2. 输入：根、层级链、请求

```python
root   = map_reduce((16, 16), (16, 16, 16))(mac, 0.0)(out=(i, j), ins=((i, k), (k, j)))(a, b)  # 循环 i, j | k
levels = serial_group((16, 16, 0), tail=1, promote=0,
           group_warp((16, 8, 0), tail=1,
             warp_red((0, 0, 8), tail=1, promote=0b11,
               warp_mma((16, 8, 8)))))
result = tile_req(root, members=(), levels, default_control)   # 恒等请求，语义 = root
```

层级链是五个互相递归的名义类型（`tensor.mim` 的 `nom TileLevels` 起，约 2442 行），类型参数保证作用域合法：

```python
TileLevels      = Done | Par(sizes, tail, rest) | Group(sizes, tail, promote, rest: GroupLevels)
                | Red(sizes, tail, promote, rest) | Vec(sizes, rest) | Cluster(size, multicast, split, rest) | Persistent(ctas, rest)
GroupLevels     = Done | Par | Item(sizes, tail, rest: ItemLevels) | Red(sizes, tail, promote, pipeline, rest)
                | RedItems(sizes, tail, merge, rest) | Vec | Warp(sizes, tail, rest: WarpLevels) | Warpgroup(sizes, tail, rest: WarpgroupLevels)
ItemLevels      = Done | Par | Red(sizes, tail, promote, pipeline, rest) | Vec | Lanes(sizes, tail, rest)
WarpLevels      = Done | Red(sizes, tail, promote, pipeline, rest) | Mma(shape, operands, block)
WarpgroupLevels = Done | Red | Wgmma(shape, operands, owned) | RedTma(...) | RedWs(sizes, tail, promote, pipeline, pingpong, consumers, rest)
```

`sizes#d = 0` 表示该级不切循环 `d`；`promote` 的第 `i` 位把输入 `i` 暂存进共享内存；`pipeline` 是在飞块数（0 取 `-X tensor:pipeline`）。

## 3. `tile_req` 实现一：定律解释器（`tensor.mim` 的 `mod law`）

### 3.1 定律与三条切分引理

`law.Law` 是 16 个字段的记录（约 3327 行），由 `build`（约 3395 行）在根的输入可见时一次造好；其中三个字段就是三条切分引理：

```python
law = Law(ro, inputs, extents=Sr, root,
          par     = λ ts:          tiled(So, ts) is (λ off, ext, xs: tile_of(make(xs), off, ext)),         # 并行拼接
          red     = λ loop, ks:    tf_reduce(K, ks, tail=1)(init_tile)(is)(step=fold_block)(last=tail_block)(post),  # 按序折叠
          red_par = λ loop, parts: post(fold(merge, [partial(j) for j in range(parts)]))  if not ordered      # 带合并的乱序折叠
                                   else red(loop, chunk),
          schedule = λ levels: tile_step(lo=0, ext=So, size=Sr, xs=phys, acc=splat(So, init), fresh=True, cx=none, levels),
          traffic, combining, shared=staged_of, phys, mean, merge, ordered, checked)

law.run(law, levels, _)  = check.report(verdict_of(law, levels), law.schedule(levels))   # 约 4203 行
law.req(law, m, levels, c) = check.report(verdict_of(...), tile_req(root, m, bitcast(levels), c))
```

`law.run` 是普通 Mim 函数：在源程序里写它时 `optimize` 的 PE 就把 `schedule` 展开成实现原语；`TileFuse` 把 `tile_req` 改写成 `law.run` 后同样展开。

### 3.2 解释器：层级 → 步骤

每个层级被解释成一个 `Step`：`(lo, ext, size, xs, acc, fresh, cx) → Tensor`，`lo`/`size` 是当前块在全部 `Rn` 个循环上的原点与尺寸，`cx = (pan, gx)` 携带面板计划与外层 work-group 的块/tile/作用域（约 3443–3462 行）。

```python
def tile_step(l):                          # 约 3830 行
    match l:
        Done                    -> leaf
        Par(s, tl, r)           -> split_par(s, tl, scope=0, k=tile_step(r))
        Group(s, tl, gp, r)     -> split_par(s, tl, scope=1, gp=gp, k=group_step(r))     # gp≠0：整块驻留（plan_whole）
        Red(s, tl, pr, r)       -> split_pro(s, tl, pr, k=tile_step(r))
        Vec(s, _)               -> leaf_vec(s)
        Cluster(...)/Persistent -> 只把参数写进 gx，继续 tile_step(r)
def group_step(l):                         # 约 3818 行
        Par -> split_par(scope=0); Item(s, tl, r) -> split_par(s, tl, 2, k=item_step(r))
        Red(s, tl, pr, pl, r)   -> split_pro(s, tl, pr, depth=pl, mma=mma_group(r), k=group_step(r))
        RedItems(s, tl, mg, r)  -> split_bound(s, tl, level=2, merge_of(mg), k=group_step(r))   # 乱序 + 合并
        Warp(s, tl, r)          -> split_par(s, tl, 3, k=warp_step(r))
        Warpgroup(s, tl, r)     -> split_par(s, tl, 6, sp=ws_wg(r), k=wg_step(r))        # sp=(是否 RedWs, pingpong)
def item_step(l):  Par -> split_par(0); Red -> split_pro(..., mma=False); Vec -> leaf_vec
                   Lanes(s, tl, r) -> split_bound(s, tl, level=4, (False, merge))
def warp_step(l):  Red -> split_pro(..., mma=mma_warp(r)); Mma(shape) -> mma(Rn, shape)(leaf)
def wg_step(l):    Red -> split_pro(tma=False); RedTma/RedWs -> split_pro(tma=True); Wgmma(shape) -> mma(Rn, shape)(leaf)
```

`split_pro` 是 `split_bound(level=0, (False, merge), promote, tiles=next_*(rest), shared, mma, depth, tma)`；`next_*` 取链中下一个并行层的 tile，面板就切得与它一样宽。

### 3.3 `split_par`：并行拼接（约 3693 行）

```python
def split_par(sizes, tail, scope, sp, gp, k):
    def step(lo, ext, size, xs, acc, fresh, cx):
        ts = [sizes[d] or ext[d] for d in range(Ro)]
        def body(off, ext2, xs2, acc2=None):
            lo2   = lo + off (仅前 Ro 维);  size2 = ext2 (前 Ro 维) | size (归约维)
            cx2   = enter(scope, sizes, lo2, size2, cx)          # scope 1 记下 work-group 块，scope 2 记下 work-item tile
            if gp: cx2.pan = plan_whole(gp, lo2, size2, cx2.pan) # Group 级 promote：整块暂存
            acc3  = splat(ext2, init) if fresh else slice(acc2, off, ext2)
            return k(lo2, ext2, size2, xs2, acc3, fresh, cx2)
        region = wrap(scope, sp, cx.gx)(body)                   # bind scope ∘ specialize ∘ cluster ∘ persistent
        return tiled(ext, ts, tail)(xs)(region) if fresh else tiled(ext, ts, tail)((xs, acc))(region)
    return step

def wrap(scope, sp, gx):  # 约 3687 行
    b = specialize(sp.pingpong)(b) if sp.ws and scope == 6
    b = cluster(...)(b) if scope == 1 and gx.cluster;  b = persistent(gx.ctas)(b) if scope == 1 and gx.ctas
    return bind(scope)(b) if scope != 0 else b
```

### 3.4 `split_bound`：按序折叠，或绑定后乱序折叠（约 3716 行）

```python
def split_bound(sizes, tail, level, (explicit, m), promote, tiles, shared, mma, depth, tma, k):
    def step(lo, ext, size, xs, acc, fresh, cx):
        loop = 第一个 sizes≠0 的归约维;  if none: return k(...)   # 该级不切归约
        K, unit = size[loop], sizes[loop]
        def block(a, k0, n, xs2):
            blo, bsize = lo 在 loop 上加 k0, size 在 loop 上取 n
            wide  = cx.gx.scope in {2, 3, 6}                  # 在 work-item / warp / warpgroup 内：面板覆盖整个 work-group 块
            width = (cx.gx.group_tiles if scope in {3, 6} else cx.gx.item_tiles) if wide else tiles
            pan   = plan(promote, width, shared, plo, psize, cx.gx, mode=0|1(mma)|2(wgmma), cx.pan) if promote else cx.pan
            return k(blo, ext, bsize, xs2, a, False, (pan, cx.gx))
        bound = fresh and level != 0                          # 只有种子累加器可以分给 work-item/lane 各自折叠
        def fold_step(a, k0):
            body = λ xs2: block(a, k0, unit 或最后一块的余量, xs2)
            body = fill_tma(body) if tma
            body = pipeline(ring)(body) if (ring := depth or check.depth()) > 1
            body = merge_with(m)(body) if explicit
            return bind(level)(body) if bound else body
        full = tile_fold(count=K/unit 或 ceil, unit)(acc)(xs)(fold_step)
        return block(full, K//unit*unit, K%unit, xs) if tail == 1 and K % unit else full   # 剥离尾块
    return step
```

### 3.5 面板计划、值与叶子

```python
def plan(promote, tiles, shared, lo, size, gx, mode, pan):        # 约 3583 行，逐输入
    for i in inputs:
        (found, axis, tile) = panel_of(maps[i], Sis[i], tiles)       # 下一并行层以单位斜率驱动的那一维
        if bit(promote, i) and found:
            box   = affine.image(maps[i], lo, size)                 # 该块读输入 i 的盒
            perm  = 外层轴顺序（shared 时交换最后两维）
            lay   = (inside_group, store=stage_shape(...), inner, vector_width, mode, reads, counts, steps, ...)
            pan[i] = ((True, axis, tile, box.lo, box.ext, perm), lay)
    return pan

def values_of(xs, ys, items, pan):                                  # 约 3493 行，叶子读的值
    for i in inputs:
        box = srcs[i].tile(ys[i], q.off, q.ext)                      # 普通输入 = slice；融合输入 = 现算生产者的盒（§5）
        if split_of(pan, i):                                         # 归约层的面板
            stg = stage(q.ext)(axis, tile, perm, ⊥)(scope=0)(box)
            lay = tc_layout.wgmma if mode == 2 else tc_layout.mma if mode == 1 else panel_layout(...)   # 布局搜索
            yield placed(store, lay)(stg) if inside_group else stg
        elif q.staged_whole:                                         # Group 级 promote：整块驻留
            yield stage(q.ext)(box 或 驻留成员的 held)
        else: yield xs[i] (普通) | box (成员)

def leaf(lo, ext, size, xs, acc, _, cx):                            # 约 3645–3653 行
    return map_reduce_post(ext, size)(f, outs=acc, id)(tile_shift(map_out, lo))(reads_of(lo, pan))(values_of(...))
```

`reads_of` 把叶子的读映射改写到面板坐标：外层轴按 `perm`，面板轴取 `c[axis] floordiv tile`，最后一维取 `c[axis] rem tile`。
`leaf_vec`（`vec_at`）把 size 1 的并行维记为向量维，并在最后一个归约维能整除时拆出展开的内层循环。

### 3.6 层级到引理的对照

| 层级 | 解释成 | 引理 |
|---|---|---|
| `Par`/`Group`/`Item`/`Warp`/`Warpgroup` | `split_par` → `tiled … (bind s …)` | `par`：并行拼接 |
| `Red`/`RedTma`/`RedWs`（未绑定） | `split_bound(level=0)` → `tile_fold` | `red`：按序折叠（+ 剥离尾块 + epilogue） |
| `RedItems`/`Lanes` | `split_bound(level=2/4)` → `tile_fold … (bind 2/4 …)` | `red_par`：各自从种子折叠，再用 `merge` 合并（需定律担保 merge） |
| `Done`/`Mma`/`Wgmma` | `leaf`（`mma` 标记） | 叶子块折进 tile |

## 4. `tile_req` 实现二：C++ `TileFuse`（`tile_fuse.cpp`）

```python
def realize(request):                                        # 约 1857 行
    (op, members, levels, control) = request
    specs = tile_levels(levels)                               # 字面层级链 → [LevelSpec(kind, bind, sizes, tail, promote, pipeline, tma, ws, pingpong, merge)]
    unsafe = 非字面 sizes
          or 某归约级 bind∈{2,4} 且无显式 merge
          or 某归约级切了非最外归约维
          or 某成员被组外使用
          or (有 Group 级 and 某成员自身带归约)
    if not unsafe and tile-law (默认 1):
        if law := law_of(op, members): return law.run(law, levels, control)   # 走 §3
    if specs[0].kind != PAR: specs.insert(0, Par(sizes=0…, tail=1))
    return Levels(...).tile(op, specs, depth=0)
```

`LevelSpec.kind`：0 并行、1 归约、2 向量、3 MMA、4 cluster、5 persistent；`bind`：1 work-group、2 work-item（含 `RedItems`）、3 warp、4 lane、6 warpgroup（表见 `tile_levels`，约 181 行）。

```python
class Levels:
    def tile(anchor, levels, depth):                          # 约 1180 行
        level, rest = levels[0], levels[1:]
        if level.kind == 5: persistent_ = level.ctas; return tile(anchor, rest)
        if level.kind == 4: cluster_ = (...);         return tile(anchor, rest)
        match level.kind:
            0: return parallel(anchor, sizes, peel, promote, bind, rest)
            1: tma_ = level.tma; return reduction(anchor, sizes, peel, promote, bind, merge, pipeline, rest)
            2: return vector(root, sizes, rest)
            3: return wgmma(root, shape) if bound_ == 6 else mma(root, shape)   # 只在 warp(3)/warpgroup(6) 内

    def parallel(anchor, sizes, ...):                         # 约 1220 行
        检查根把循环 d 写到输出轴 d（canonical_write）
        if 整块且不绑定不 promote: return tile(anchor, rest)
        promote 的输入：stage_whole（整块进共享内存）
        tile_var = 新区域的 (off, ext, ins)
        fused = fuse(root_tile, var, depth, kind=0)           # 问 control 是否把成员的 tile 拉进区域
        value = tile(fused.value, rest, depth + 1)
        region = λ off, ext, ins: value
        region = specialize / cluster / persistent / bind(bind)(region)
        return tf_par(So, ts, tail)(ins)(region)  或 tf_par_n(...)（有成员要交还到区域外时）

    def reduction(anchor, sizes, ...):                        # 约 1455 行
        loop = 唯一被切的归约维（并行维不许切）
        检查：内层归约维乱序需可重结合的 combiner；lane 归约需 2 的幂个 lane 等
        init    = tile_of_init(root)
        partial = tile_of_partial(root, loop, ko, tf_chunk(K, ks, tail, ko), acc)
        fused   = fuse(partial, var, depth, kind=1); fused = stage_block(fused, ko, chunk)   # 面板 stage+placed
        value   = tile(fused.value, rest)
        step    = λ (acc, ko): bind?(merge_with?(pipeline?(fill_tma?(λ ins: value))))
        last    = 剥离的尾块（同样 fuse/stage/tile）或恒等
        return tf_reduce(K, ks, tail)(init)(ins)(step)(last)(post)   # = tile_fold + last + post

    def mma(root, shape):  检查 (16,8,8)/(16,8,16)、操作数已暂存且 A 行主序 → tensor.mma(Rn, shape)(root)
    def wgmma(root, shape): 检查 16 位输入、(64, 8n≤256, 16)、覆盖整个 warpgroup tile → tensor.mma(Rn, shape)(root)
```

两条实现产生同一组原语（`tiled`/`bind`/`tile_fold`/`stage`/`placed`/`pipeline`/`fill_tma`/`specialize`/`mma`），因此后续阶段不区分它们。

## 5. 融合

```python
# C++ 入口（约 1805 行）：成员可以嵌套，每个被根直接读的成员变成一个“来源定律”
def law_of(op, group, merge):
    for i, x in enumerate(op.inputs):
        ins[i] = law.source(x) if x not in group else law.as_source(law_of(x, group, vouched(x)))
    if no member: return of_map_reduce(…)(op.inputs)  或 of_map_reduce_merging
    return of_sources(…)(Ps, ins)(op.inputs)                   # 融合了生产者的根是有序的

# Mim 侧（约 4027–4120 行）
fuse(slot, producer…)(consumer…)          = fuse_law(slot, of_map_reduce(producer…))(consumer…)
fuse_law(slot, pl)(consumer…)             = build(consumer…, srcs) where srcs[slot] = Src(
        phys = pl.phys,                                        # 区域线程化生产者的物理输入，而非它的结果
        whole = pl.mean,  tile = λ p, off, ext: tile_of(pl.mean(p), off, ext),   # 块只现算它读的盒
        merge = pl.merge, split = (pl 担保 merge 且只多一个归约维), K = pl 的最后一维, fused = True)
```

`values_of` 里融合输入的盒由 `tile_of(producer, off, ext)` 给出，PE 后就是生产者在该盒上的 `map_reduce_post`，再被 `stage`/`placed` 成面板（原文 §8 中 `dequant` 的例子）。
驻留成员（Group 级 promote 的融合输入）在 work-group 内整块暂存为 `stage(E)(held)`：`held` 默认逐元素 `generate(E, λ c: tile_of(producer, off + c, 1))`；当生产者可拆（`split`）且 work-item 多于元素数 `n` 时，取 `C = items / n`（需整除 `K`），先 `generate` 出 `E × C` 个部分和 `tile_of_partial(…, k0 = j·K/C, K/C, seed)`，再逐元素用 `merge` 合并（`values_of` 中的 `each`/`joint`）。

`TileFuse::realize` 目前对“有 Group 级且成员带归约”的请求退回 C++ 路径（`Levels::fuse` + `split_over_items`）；直接写 `law.run` 的源程序可走上面的 `joint` 分支。

## 6. `tensor.lower_to_mem`（`lower_to_mem.cpp`）

```python
def rewrite_imm_App(app):                                     # 约 682 行
    mma / pipeline / fill_tma / frozen      -> 透传（由外层按标记处理）
    tiled / tiled_multi                     -> lower_tiled
    tile_fold                               -> lower_tile_fold
    stage                                   -> lower_stage（→ stage_impl → generate → lower_generate）
    placed(stage)                           -> placed_layouts_[stage] = layout；透传
    slice                                   -> slice_impl
    map_reduce_post                         -> lower_map_reduce
def lower_into(root, dst):                                    # 带目的缓冲区的版本，约 1461 行
    map_reduce_post -> store_epilogue 或 lower_map_reduce(dst);  mma -> lower_mma(dst, fragments) 若 dst 是 MMA 片段
```

### 6.1 `lower_tiled`（约 1636 行）

```python
def lower_tiled(tiled(ext, ts, tail)(ins)(region)):
    count = Π ceil(ext[d] / ts[d])
    if region 无 bind: return for t in range(count): body(t)                 # affine.For "tileGrid"，单 tile 不生成循环
    tag = {1: group, 2: item, 6: specialized if specialize else warpgroup, 7: warpgroup}.get(bind, warp)
    launched = count + (1 if specialize)                     # 多一个生产者（outline 中只跑一个 warp）
    if persistent(ctas): for round in tileRounds: tile = ctas*round + id; skip if tile >= count
    if specialize: state.mem = reg_budget(min(255, 16384 / (32 * per_partition) // 8 * 8))(state.mem)
                   producer = (linear == count); linear %= count
    return for t in range(launched) @tag: lower_tile_body(t)                # 一维循环，outline 映射到线程号
```

work-group 的结果缓冲区放在 `shared_scope`；被 work-item 协作写入的 work-group 缓冲区在之后加 `fence`。

### 6.2 `lower_stage` / `lower_generate`：面板填充

```python
def fill_panel(stage):                                        # lower_generate 约 1033–1080 行
    if batch_ and uniform and 非流水 and 非 async:             # 同步面板：加入消费者的“批”
        if batch_.token is None:
            batch_.token = enter_write(g, 0)(group_enter(g)(start_mem))
        p = shared_panel(g, 0)(shape)(batch_.token)
        token, buf = coop_fill(g, 0)(p, panelFill = for e in elems @chained: buf[e] = src[e])
        return view(publish_view(batch_.published, buf))       # published 在 close_batch 时补上
    else:
        dst = ring 槽（流水）或 alloc(shared_scope)
        for e in elems @(async if async_fill_ else coop): dst[e] = src[e]    # coop 前后各一个屏障
def close_batch(mem, consume):                                # 约 547 行
    published = sync_publish(g, 0)(token)
    out = read_phase(g, 0)(published, batchRead = consume)
    return leave_reading(g, 0)(out.mem), out.value
```

`open_batch` 在 `lower_map_reduce`（约 1300 行）与 `lower_mma` 入口打开批，故一个叶子读的所有同步面板共享一个“进入—填充—发布—读—离开”超步。

### 6.3 `lower_tile_fold`（约 3613 行）

```python
def lower_tile_fold(tile_fold(n, ks)(acc)(ins)(step)):
    (body, level, merge, depth, tma) = peel_step(step)
    if level == 2: return lower_bound_fold(...)              # §6.5
    if level == 4: return lower_lane_fold(...)               # lane 分摊 + shfl 合并
    panels = 带 pipeline 标记的 stage;  ring = [[alloc(shared_scope) for p in panels] for s in range(depth)]
    specialized = tma and producer_ and fragments.group       # RedWs
    if tma:
        full[s]  = mbar_init_if(count=1, leader)  for s in range(depth)
        empty[s] = mbar_init_if(count=4*consumers, leader) if specialized or cluster 共享
        barrier()
    if specialized and pingpong: arrive_named_if(bar 1, 256, 第二个消费者)
    if not specialized:                                      # 序言：填前 depth-1 块
        for s in range(depth - 1): fill(s, ring[s])
    for i in range(n) as tileFoldStep:
        if specialized:
            if producer:                                     # foldProduce
                mbar_wait_if(empty[i % depth], empty_parity(i, depth), issuer)
                fill_tma(block i, ring[0])
            else:                                            # foldConsume
                mbar_wait_at(full[i % depth], full_parity(i, depth))
                acc = lower_into(body, acc)                  # → lower_mma（wgmma，wgmma_wait(pending=overlap)）
                mbar_arrive_if(empty[released], lane == 0)   # released = i % depth，或 depth≥4 时上一步的槽
        else:
            if panels:
                if tma: mbar_wait_at(full[i % depth], full_parity(i, depth))
                else:   async_wait_group(depth - 2); fence_proxy_async() if wgmma
                barrier()
                fill(refill_block(i, depth, n) = min(i + depth - 1, n - 1), ring[depth - 1])
            acc = lower_into(body, acc) / copy 回累加器
        ring = rotate(ring)                                  # ring[s] ← ring[(s+1) % depth]
    async_wait(); (TMA 时等最后 depth-1 次多余的填充); wgmma_wait(0) if specialized

def fill(block, slots):                                      # cp.async 版：fill_block
    for p in panels: for e in elems @async: slots[p][e] = src[p][block, e]
    async_commit()
def fill_tma(block, slots):                                  # TMA 版
    mbar_expect_tx_if(full[slot], bytes, issuer)
    for p in panels: for 每个 128 字节宽的原子列 a: tma_load_if(slots[p] + a, tensor_map[p], col0 + a, row0, full[slot], issuer)
```

`issuer` 在非特化时是 `leader()`（tid == 0），特化时是“生产者且 `item & 127 == 0`”；`buf_tensor_map` 在 kernel 外构造。

### 6.4 `lower_mma`（约 2576 行）

```python
def lower_mma(mma(n, shape)(leaf), dst, fragments):          # 在 open_batch/close_batch 之间
    if fragments.group:                                      # wgmma：一个 warpgroup 的整个 64×N tile
        if pingpong: sync_named(own_bar, 256)
        wgmma_fence(publish_panels())
        for ks in range(depth // 16):
            acc = wgmma_f16/bf16(nt=N/8, desc(A, ks), desc(B, ks), b_mn, acc)   # desc 由面板地址按位拼出
        wgmma_commit()
        if pingpong: arrive_named(other_bar, 256)
        acc = wgmma_wait(pending)(acc)
    else:                                                    # mma.sync：每个 warp
        for ks in range(depth // kt):                         # kt = 8 (TF32) 或 16 (F16/BF16)
            for mi in range(m // 16): a[mi] = buf_ldmatrix(x4, A_panel 的 lane 地址); tf32_rna 若 F32
            for ni in range(n // 8):  b[ni] = 按 lane 读 B_panel 的 2 个元素
            for mi, ni: acc[mi, ni] = mma_tf32/f16/bf16(a[mi], b[ni], acc[mi, ni])
    return acc   # 片段；store_epilogue 按 m16n8 C 布局写回（行 lane/4 与 lane/4+8）
```

其余叶子由 `lower_map_reduce`（约 1242 行）处理：`open_batch` → 重写输入（触发面板填充）→ `close_batch(λ m: btensor.map_reduce_post(…))`，循环留给后面的 `btensor.lower_map_reduce` 生成。

### 6.5 `lower_bound_fold`（约 4301 行）：`RedItems` 的 work-item 分摊

```python
token = enter_write(g, ep)(group_enter(g)(start))
partials = shared_panel(token)                                # 所有 work-item 的部分和沿第一维堆叠
coop_fill(partials, boundFoldFill =
    for it in boundFoldItems @item:
        own = seed; own = fold(block it)                      # 每个 work-item 从种子折叠自己的块
        own = warp 内 shfl_xor 归并（可选）
        partials[it] = own)
token = sync_publish(g, ep)(…);  view = publish_view(token, partials)
if items 是 2 的幂:                                           # 树形归并，每轮一个超步
    for round: token = enter_update(token); owned_update(claim(view), boundTreeRound); sync_carry(); enter_read()
else: 每个元素一个 work-item 按序合并
out = read_phase(g, at)(boundCombine);  sync_reclaim(g, at);  group_leave(g, ep + 1)
```

## 7. `lower_to_mem` 之后的公共阶段

### 7.1 `gpu.check_sync` / `gpu.erase_phases`

```python
# erase_phases.cpp（48 行）
GM(g, e, p)                               -> mem.M 0;   SharedBuf(...) -> buffer.Buf
group_enter / sync_publish / sync_reclaim -> fence(mem)
enter_write / enter_read / enter_update / group_leave / leave_reading / view -> 透传
shared_panel(g, e)(r, s, T)(mem)          -> buffer.alloc(r, s, T)(panel_scope(g)(mem))
coop_fill(mem, buf, fill)                 -> fill(mem, buf);   publish_view -> buf;   claim -> buf
owned_update(mem, buf, body)              -> (body(mem, buf), buf);   sync_carry -> (fence(mem), buf)
read_phase(mem, body)                     -> body(mem)
```

`check_sync` 先检查令牌恰好用一次、不越出生成它的函数、构造不重入；它不改 IR。
`buffer.lower_ptr` 随后把 `Buf(r, s, T)` 变成 `Ptr(«s; T»)`、`read`/`write` 变成 `lea` + `load`/`store`、`buf_ldmatrix` 变成 `ldmatrix`、`buf_addr` 变成 `lea` + `shared_addr`。

### 7.2 `gpu.outline`（`outline.cpp`）

```python
def rewrite(app):                                            # 约 100–250 行
    for g in range(G) @group:  body        -> kernel tileKernel(args, group_id, item_id) { body }
                                              host: alloc + copy_to_device; launch(G, n_items); copy_to_host
    for w in range(W) @item/warp/warpgroup/specialized:
        idx  = tid (item) | tid >> 5 (warp) | tid >> 7 (warpgroup/specialized)
        lane = tid & 31;  rank = (tid >> 5) & 3
        if W*width > n_items: for turn: idx = tid + turn * n_items          # 多于 1024 个时轮转
        elif W*width < n_items: if tid < W*width: body else skip            # 多余线程跳过
        else: body[w := idx]
    for e in range(E) @coop:    barrier(); for e in range(tid, E, n_items): body; barrier()
    for e in range(E) @chained: for e in range(tid, E, n_items): body               # 无屏障，由 sync_publish 发布
    for e in range(E) @async:   同上，且 body 中的 store → store_async                # cp.async
    alloc(shared_scope | panel_scope(g))  -> slot_shared(T)                         # 静态槽；不同构造的面板可复用同一区域
    fence(m)                               -> sync_work_items(m)
    lane_id / item_index / leader / warp_rank -> lane / tid / tid == 0 / rank
```

### 7.3 CPS 化与 `nvptx.emit`

```python
affine.lower_index; cps.conv; affine.lower_for (affine.For → 递归函数); 闭包转换 ...
split_apply: kernel → 设备世界
DeviceEmitter:  sync_work_items -> llvm.nvvm.barrier0 (bar.sync 0)
                store_async     -> llvm.nvvm.cp.async.{ca,cg}.shared.global.{4,8,16}
                async_commit / async_wait_group -> cp.async.commit_group / wait_group
                ldmatrix / mma_tf32 / mma_f16 -> llvm.nvvm.ldmatrix.* / llvm.nvvm.mma.m16n8k8.row.col.tf32 / m16n8k16
                wgmma_fence / commit / wait -> llvm.nvvm.wgmma.*
                wgmma_f16 / mbar_* / tma_load_if / setmaxnreg -> 内联 PTX（谓词 @q 来自 issue 参数）
                slot_shared -> 静态 .shared 或（超过静态上限、TMA 目标）1024 对齐的动态共享内存
                reg_budget  -> "nvvm.maxnreg" 属性 (.maxnreg)
opt -passes=nvvm-reflect,default<O3>; llc -mcpu=sm_XX; ptxas; fatbinary; HostEmitter 嵌入 fatbin
```

## 8. SIMT 路径

### 8.1 同步面板：`tile_outline_panel.mim`（16×16×16，F32）

链：`serial_group((8,8,0), 0, 0, group_red((0,0,16), 1, 3, group_item((4,4,0), 0, item_vec((1,0,0), item_done()))))`。

```python
# group_default 之后
C = tiled((16,16), (8,8), 0)((A, B), bind(1)(λ g:                       # 2×2 个 work-group
      tile_fold(1, 16)(acc=splat(0))((A, B), λ (acc, k0):              # k 只有一块
          pA = placed(stage(slice(A, (g.i, k0), (8, 16))));  pB = placed(stage(slice(B, (k0, g.j), (16, 8))))
          tiled((8,8), (4,4), 0)((pA, pB), acc, bind(2)(λ t:            # 2×2 个 work-item
              map_reduce_post(4×4×16, vec dim 0)(mac)(panel reads)(pA, pB))))))

# lower_to_mem 之后
for g in range(4) @group:
    for t in range(4) @item:
        tok = enter_write(group_enter(m))
        sA, tok = coop_fill(shared_panel(tok), for e @chained: sA[e] = A[g.i + .., e])
        sB, tok = coop_fill(shared_panel(tok), for e @chained: sB[e] = B[.., g.j + ..])
        pub = sync_publish(tok)
        out = read_phase(pub, λ m: btensor.map_reduce_post(4×4×16)(view(sA), view(sB)) → C[g, t])
        leave_reading(out)

# erase_phases + outline 之后：launch(groups=4, items=4)
kernel(gid, tid):
    barrier()                                               # group_enter
    smem.A[...] = A[...] (tid 分摊, ld.global.v4 / st.shared.v4)
    smem.B[...] = B[...]
    barrier()                                               # sync_publish
    for i, j in 4×4: for k in 16: c[i,j] = fma(smem.A[i,k], smem.B[k,j], c[i,j])   # ld.shared + fma.rn.f32
    C[...] = c
```

### 8.2 cp.async 环：`tile_pipeline.mim`

链：`serial_group((8,8,0), 1, 0, group_item((2,2,0), 1, item_red_pipelined((0,0,4), 1, 3, 3, item_vec(…))))`。

```python
# group_default 之后
tiled((16,16), (8,8), 1)(bind(1)(λ g:                      # 4 个 work-group
    tiled((8,8), (2,2), 1)(bind(2)(λ t:                     # 16 个 work-item
        tile_fold(4, 4)(splat(0))(λ (acc, k0): pipeline(3)(λ:
            叶子读 placed(stage(A 盒 8×4)), placed(stage(B 盒 4×8))))))))   # 面板覆盖 work-group 块

# lower_to_mem → outline → PTX
kernel(gid, tid):                                          # launch(4, 16)
    ring = smem[3][2]                                        # 3 级 × 2 个输入，共 6 个 128 字节槽
    for s in 0, 1: for e in elems(tid) : cp.async(ring[s][·][e] ← block s); commit_group
    for i in range(4):
        cp.async.wait_group(1)                               # pending = depth - 2
        barrier()
        for e in elems(tid): cp.async(ring[2][·][e] ← block min(i + 2, 3)); commit_group
        for k in 4: c += ring[0].A[·, k] * ring[0].B[k, ·]   # ld.shared + fma
        ring = rotate(ring)
```

## 9. MMA 路径：`tile_mma.mim`（`-X tensor:pipeline=2`）

链：`serial_group((16,16,0), 1, 0, group_warp((16,8,0), 1, warp_red((0,0,8), 1, 3, warp_mma((16,8,8)))))`。

```python
# group_default 之后
tiled((16,16), (16,16), 1)(bind(1)(λ g:                    # 1 个 work-group
    tiled((16,16), (16,8), 1)(bind(3)(λ w:                  # 2 个 warp
        tile_fold(2, 8)(splat(0))(λ (acc, k0): pipeline(2)(λ:
            mma(3, (16,8,8))(map_reduce_post(16×8×8)(placed(stage(A 16×8)), placed(stage(B 8×16))))))))))

# lower_to_mem 之后（每个 warp；面板在 shared_scope，所有 work-item 协作填）
for g in range(1) @group:
  for w in range(2) @warp:
    ring.A = [smem(16,12)] * 2;  ring.B = [smem(8,20)] * 2      # 带填充的布局，避免 bank 冲突
    fill(block 0 → ring[0]) @async; async_commit
    acc: «4; F32» = 0                                         # 每个 lane 的 m16n8 C 片段
    for i in range(2):
        async_wait_group(0); fence()
        fill(block min(i+1, 1) → ring[1]) @async; async_commit
        a = buf_ldmatrix(x4, ring[0].A @ lane 地址)             # TF32 借用 b16 的 ldmatrix
        b = (ring[0].B[lane%4, w*8 + lane/4], ring[0].B[lane%4 + 4, …])
        acc = mma_tf32(a, b, acc)
        swap(ring[0], ring[1])
    async_wait()
    C[w 的行 lane/4 与 lane/4+8, 列] = acc                    # 两次 buffer.write

# outline → PTX：launch(1, 64)
    cp.async.cg ×…; commit_group; wait_group 0; bar.sync 0
    cp.async.cg (下一块); commit_group
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0..a3}
    ld.shared.b32 b0; ld.shared.b32 b1
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {d}, {a}, {b}, {c}
```

## 10. Hopper wgmma + warp specialization：`tile_wgmma_ws.mim`（128³，F16 输入，F32 累加）

链：`serial_group((128,128,0), 1, 0, group_warpgroup((64,128,0), 1, wg_red_ws((0,0,64), 1, 3, 2, 0, wg_mma((64,128,16)))))`。

```python
# group_default 之后
tiled((128,128), (128,128), 1)(bind(1)(λ g:
    tiled((128,128), (64,128), 1)(bind(6)(specialize(0)(λ wg:        # 2 个消费者 warpgroup
        tile_fold(2, 64)(splat(0))(λ (acc, k0): pipeline(2)(fill_tma(λ:
            mma(3, (64,128,16))(map_reduce_post(placed(stage(A 128×64), wgmma 128B swizzle K-major),
                                                placed(stage(B …), 128B swizzle)))))))))))

# lower_to_mem 之后
host: mapA = buf_tensor_map(A, box 128×64); mapB = buf_tensor_map(B, box 64×64)
for g in range(1) @group:
  for wg in range(2 + 1) @specialized:                       # 第 3 个只有一个 warp：生产者
    reg_budget(168)                                          # 16384 / (32·3) 向下取 8 的倍数
    producer = (wg == 2)
    ring.A = smem(128,64)[2]; ring.B = smem(2,64,64)[2]; full = smem«2; I64»; empty = smem«2; I64»
    for s in 0, 1: mbar_init_if(full[s], 1, tid == 0); mbar_init_if(empty[s], 8, tid == 0)   # 8 = 2 个消费者 × 4 warp
    barrier()
    for i in range(2):
        slot = i % 2
        if producer:                                         # foldProduce：只有生产者 warp 的第一个线程发
            mbar_wait_if(empty[slot], parity=(i/2 + 1) % 2, issue)
            mbar_expect_tx_if(full[slot], 32768, issue)       # 128·64·2 + 2·64·64·2
            tma_load_if(ring.A[slot], mapA, col=k0, row=row0, full[slot], issue)
            tma_load_if(ring.B[slot][0], mapB, col=0,  row=k0, full[slot], issue)   # B 按 128 字节宽的列原子分两次
            tma_load_if(ring.B[slot][1], mapB, col=64, row=k0, full[slot], issue)
        else:                                                # foldConsume
            mbar_wait_at(full[slot], parity=(i/2) % 2)
            wgmma_fence()
            for ks in range(4): acc = wgmma_f16(n=128/8, desc(A[slot] 的 wg 行, ks), desc(B[slot], ks), b_mn=True, acc)
            wgmma_commit(); acc = wgmma_wait(0)(acc)
            mbar_arrive_if(empty[slot], lane == 0)           # 每个消费者 warp arrive 一次
    async_wait(); wgmma_wait(0)
    if not producer: C[wg 的 64 行] = acc                    # store_epilogue，生产者不写

# outline → PTX：launch(1, 288)，`.maxnreg 168`，面板在 1024 对齐的动态共享内存
    @q mbarrier.init.shared::cta.b64 …; bar.sync 0
  foldProduce:
    mbarrier.try_wait.parity.acquire.cluster…  (empty)
    @q mbarrier.arrive.expect_tx.shared::cta.b64 _, [full], 32768
    @q cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes …   ×3
  foldConsume:
    mbarrier.try_wait.parity.shared::cta.b64 (full)
    wgmma.fence.sync.aligned
    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 …   ×4
    wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0
    @q mbarrier.arrive.shared::cta.b64 _, [empty]
```

变体：

- `RedTma`（`wg_red_tma`）：没有 `specialize`，`bind 6` 落成 `@warpgroup`；只有 `full` 屏障，由 `leader()` 发 TMA；每步 `mbar_wait_at(full) → barrier() → fill_tma(refill 块) → wgmma`，与 §6.3 非特化分支相同。
- ping-pong（`wg_red_ws(…, pingpong=1, …)`）：`lower_tile_fold` 开头让第二个消费者 `arrive_named_if(bar 1, 256)` 一次；`lower_mma` 每次 wgmma 序列前 `sync_named(own)`、提交后 `arrive_named(other)`，两个消费者轮流发 MMA。
- ring 深度 ≥ 4 时 `overlap = 1`：`wgmma_wait(1)` 留一组在飞，消费者释放的是上一步的槽。

## 11. 不确定之处

- （已核实）`TileFuse` 构造的 `law.run (…)` 是普通 `anx lam` 的应用，构造这个 App 时就被 β 化简；解释器里带 `core.pe.is_closed` 过滤器的具名 lam 在参数成为字面量时随即展开，所以 `group_default` 之后已是实现原语。
- §3 的伪代码省略了 `enter`/`Gx` 的完整字段语义与 `plan` 中 `counts`/`steps`/`agree` 的布局搜索细节，只写了它们的作用；`panel_layout`、`tc_layout.*` 的内部没有展开。
- §5 驻留成员的 `C` 的取法按 `values_of` 的代码概括（`C = items / n`，且需 `split` 且 `K % C == 0`，否则 1）；`TileFuse::realize` 当前对“Group 级 + 带归约成员”退回 C++，这依据的是工作区中（未提交修改的）`tile_fuse.cpp`。
- §6.3 的循环骨架省略了 `first_scalar`、cluster 拆分归约（`cluster_rank` 偏移）、persistent 的 `round_base`/`ringSetUp`、剥离尾块等分支；`lower_lane_fold` 只提了名字。
- §8.1 中 `item_vec` 的向量化、§8.2 的 work-item 数（16）与面板盒尺寸是按层级链推算的，原文没有给出它们的 `lower_to_mem` 片段。
- §9 中 B 片段的 lane 坐标按 `mma.m16n8k8` 的 TF32 布局示意，精确映射见 `fragment_coords`/`mma_m16n8k8_tf32::l_b`。
- §10 中 A 盒按两个消费者的 128 行整体暂存、各消费者以描述符偏移读自己的 64 行，这是从 `slot_shared «128, 64»` 与 `wgmma_desc` 推断的。
- §10 中 B 的 TMA 坐标（`col` 为 n 偏移 0/64、`row` 为 k0）与 `b_mn = True` 取自原文转储里 `tma_load_if` 的参数顺序和 `gpu.wgmma_f16 (16, …, tt, …)`；这与 `wg_mma` 文档“两个输入都是 K-major”的说法看似不一致，本文未进一步核实。
- `RedTma`、ping-pong 的展开只依据源码，没有转储核对；本文没有运行任何 `mim` 或 GPU 程序。
