# 从张量程序到 NVIDIA GPU：tiling 编译流水线（2026-11-16）

本文按阶段描述 MimIR 怎样把一个带多级 tiling 调度的张量程序编译成 NVIDIA GPU 代码。
每个阶段给出该阶段之后 IR 的真实片段（节选）和实现位置，最后分别讲 SIMT、MMA（`mma.sync`）、Hopper wgmma + warp specialization 三条路径在哪里分叉、各自落成什么。

所有 IR 片段都是用 `build-h200-debug/bin/mim` 实际转储的（2026-11-16，分支 `feat/gpu-tiling`），gid 后缀已由打印器重命名成 `_N`。
片段只节选能说明问题的部分，`…` 表示省略。
行号是写作时的近似值，`tensor.mim` 正在频繁修改，以函数名为准。

## 0. 怎样复现每一步的 IR

默认流水线是 `src/mim/plug/opt/opt.mim` 的 `_default_compile`。
要看某一阶段之后的 IR，在示例程序末尾追加一个只跑到该阶段为止的 `_compile`（它优先于 `_default_compile`），例如只跑到 `tensor.lower_to_mem`：

```mim
plugin compile;
plugin mem;
extern lam _compile (): compile.Phase =
    let optimize = compile.phases ff (compile.scalarize, compile.phases tt (compile.beta_red, compile.eta_conv),
                                      mem.heap2stack, mem.seo, compile.adce, compile.static_arg_opt);
    compile.phases ff (
        compile.named "tensor.raise_attention", compile.named "tensor.group_attention", optimize,
        compile.named "tensor.reassoc", compile.named "tensor.lower_tensor", compile.named "tensor.fuse_tensor",
        compile.named "tensor.group_default", compile.named "tensor.lower_to_mem",
        compile.cleanup
    );
```

然后：

```sh
mim -p opt -X tensor:tile-fuse=1 -p gpu file.mim -o -          # 打印该点的 IR
mim -p opt -X tensor:tile-fuse=1 file.mim -p nvptx -X nvptx:sm=90a   # 完整流水线，产生 file_dev.ll / file_dev.ptx
```

`-VVVV` 会打印每个 phase 的日志；`tensor.tile_fuse runs root_N as its law` 一行说明 `tile_req` 走的是定律解释器（见 §2.3）。

本文用到的示例（均为 `lit/gpu/` 下的测试，单个 `extern`，16×16 或 128×128 的 GEMM）：

| 路径 | 示例 | 层级链 |
|---|---|---|
| SIMT（同步面板） | `tile_outline_panel.mim` | `serial_group ((8, 8, 0), 0, 0, group_red ((0, 0, 16), 1, 3, group_item ((4, 4, 0), 0, item_vec ((1, 0, 0), item_done ()))))` |
| SIMT（cp.async 环） | `tile_pipeline.mim` | `serial_group ((8, 8, 0), 1, 0, group_item ((2, 2, 0), 1, item_red_pipelined ((0, 0, 4), 1, 3, 3, item_vec (…))))` |
| MMA | `tile_mma.mim`（`-X tensor:pipeline=2`） | `serial_group ((16, 16, 0), 1, 0, group_warp ((16, 8, 0), 1, warp_red ((0, 0, 8), 1, 3, warp_mma (16, 8, 8))))` |
| wgmma + WS | `tile_wgmma_ws.mim` | `serial_group ((128, 128, 0), 1, 0, group_warpgroup ((64, 128, 0), 1, wg_red_ws ((0, 0, 64), 1, 3, 2, 0, wg_mma (64, 128, 16))))` |
| wgmma + TMA（无特化） | `tile_wgmma_tma.mim` | 同上，`wg_red_tma ((0, 0, 64), 1, 3, 2, …)` |
| 融合 | `law_fuse.mim` 中的 `simt_run` | `law.fuse 1 …` + SIMT 链 |

## 1. 总览

### 1.1 三层分工

| 层 | 回答什么 | 在 IR 里是什么 | 实现 |
|---|---|---|---|
| 语义（定律） | 这个计算能怎样切、切完怎样合 | `law.Law` 记录：`ro`、`extents`、`root`、`par`/`red`/`red_par`、`merge`、`schedule` 等字段 | `tensor.mim` 的 `mod law`（`of_map_reduce`、`fuse`、`fuse_shared`、`fuse_law`） |
| 调度（层级链） | 在哪一级、多大、放哪、何时 | `TileLevels` / `GroupLevels` / `ItemLevels` / `WarpLevels` / `WarpgroupLevels` 的值，类型以作用域、tile 形状、收缩种类为索引 | `tensor.mim` 从 `nom TileLevels` 起（约 2442 行）的构造子 `serial_group`、`group_red`、`warp_mma`、`wg_red_ws` 等 |
| 实现 | 降成循环、内存、同步、指令 | `tensor.tiled`/`tile_fold`/`bind`/`stage`/`pipeline`/`mma` → `buffer.*` + `gpu.*` → `mem.*` → LLVM/PTX | `tensor.lower_to_mem`（C++）、`gpu.*` phase、`nvptx.emit` |

调度通过两条途径进入 IR：

- `tensor.tile_req (root, members, levels, control)`：一个恒等请求，`tensor.group_default`（`-X tensor:tile-fuse=1` 时）把它兑现。
- `law.run (law, levels, control)` / `law.req (…)`：直接作用在定律上；`law.run` 是普通 Mim 函数，在通用优化（PE）时就被展开成实现原语。

### 1.2 `tile_req` 的两条实现

`TileFuse::realize`（`src/mim/plug/tensor/phase/tile_fuse.cpp`，约 1847 行）对每个请求二选一：

1. **定律解释器**（默认）。`TileFuse::run_as_law`（约 1808 行）检查根是“朴素” `map_reduce`（无 epilogue、字面形状、规范 schedule，`plain_parts`），最多一个成员；满足时构造 `law.of_map_reduce …`（有成员时是 `law.fuse slot …`），然后改写成 `law.run (law, levels, control)`，由 `tensor.mim` 中 `mod law` 的 `build` 里的 `tile_step`/`group_step`/`item_step`/`warp_step`/`wg_step` 按层级逐级解释（约 3801–3846 行）。
   `-X tensor:tile-law=0` 关掉这一条；遇到解释器尚不支持的情况（非字面层级尺寸、并行嵌套的归约切分、成员被组外使用、work-group 内带归约的驻留成员等，见约 1860–1885 行的 `unsafe` 判断）也会自动退回第二条。
2. **C++ TileFuse**：`Levels::tile`（约 1180 行）逐级递归，`Levels::parallel`、`Levels::reduction`、`Levels::mma`、`Levels::wgmma`、`Levels::vector` 分别处理各类层级，并做融合成员的 `Levels::fuse`。

两条实现产生**同一组实现原语**。对 `tile_mma.mim`，`-X tensor:tile-law=0` 与默认输出中出现的 `tensor.*` 原语完全相同（`tiled`×2、`bind`×2、`tile_fold`、`pipeline`、`mma`、`stage`×2、`placed`×2、`slice`×2、`splat`、`map_reduce_post`），只差 C++ 路径多一个 `tensor.clip_sched`。
因此 `group_default` 之后的所有阶段不区分两条实现。

### 1.3 阶段表

下表只列与 tiling/GPU 相关的阶段（`opt.mim` 中其余阶段如 `regex.*`、`autodiff.*`、`clos.*` 对这些示例是空操作）。

| 阶段 | 输入 IR | 输出 IR | 实现文件 |
|---|---|---|---|
| `tensor.raise_attention`、`group_attention` | 张量算子图 | 识别出的 attention 变成 `sdpa` 等（GEMM 示例不受影响） | `tensor/phase/group_rules.cpp` |
| 通用优化 `optimize` | 带 `law.run` 的程序 | `law.run` 被 PE 展开为 `tiled`/`tile_fold`/…（`tile_req` 不动） | `compile`、`mem` 插件 |
| `tensor.reassoc` | 矩阵链 | 重结合后的链 | `tensor/phase/reassoc.cpp` |
| `tensor.lower_tensor` | 派生张量算子 | 核心算子 `map_reduce_post` 等 | `tensor/phase/lower.cpp` |
| `tensor.fuse_tensor` | 核心算子图 | 生产者并入消费者的读映射、epilogue 并入 `post` | `tensor/phase/fuse.cpp` |
| `tensor.group_default` | `tile_req (root, …, levels, …)` | `tensor.tiled` / `bind` / `tile_fold` / `stage` / `placed` / `pipeline` / `fill_tma` / `specialize` / `mma` 嵌套的区域 | `tensor/phase/group_rules.cpp`（`GroupDefault::start`）→ `tile_fuse.cpp` |
| `tensor.lower_to_mem` | 上一行的张量区域 | `buffer.Buf` + `affine.For` + `gpu.bind.*` 循环；GPU 原语 `gpu.mma_tf32`、`gpu.buf_ldmatrix`、`gpu.wgmma_f16`、`gpu.tma_load_if`、`gpu.mbar_*`、`gpu.async_*`；同步面板用工作组阶段令牌 `gpu.group_enter`/`coop_fill`/`sync_publish` | `tensor/phase/lower_to_mem.cpp`（4875 行） |
| `gpu.check_sync` | 带阶段令牌的 IR | 不变；检查令牌与面板的线性使用 | `gpu/phase/check_sync.cpp` |
| `gpu.erase_phases` | 阶段令牌 | `gpu.fence`、`gpu.panel_scope`、普通 `buffer.Buf` | `gpu/phase/erase_phases.cpp` |
| `tensor.distribute` | warp 值（`«32; T»`）、`mma_warp`、`warp_keep` | 每个 lane 的分量、`shfl_*` | `tensor/phase/distribute.cpp` |
| `mem.add_mem` | 隐式内存 | 显式 `mem.M` 链 | `mem` 插件 |
| `gpu.lower_btensor_map_reduce`、`btensor.lower_map_reduce` | 剩余的 `btensor.map_reduce_post`（叶子计算） | 嵌套 `affine.For` | `gpu/phase/lower_map_reduce.cpp`、`btensor` 插件 |
| `gpu.mem_checks` | — | 检查 `mem.M` 不跨设备边界 | `gpu/phase/mem_checks.cpp` |
| `buffer.lower_ptr` | `buffer.Buf`/`read`/`write`/`alloc` | `core.Ptr` + `mem.lea`/`load`/`store`；`buf_ldmatrix`→`ldmatrix`，`buf_addr`→`shared_addr` | `buffer` 插件 |
| `gpu.outline` | `gpu.bind.group` 循环 | `con tileKernel …` + 主机侧 `gpu.launch`、`copy_to_device`/`copy_to_host`；`bind.item/warp/warpgroup/specialized` 变成线程号算术；`shared_scope`→`slot_shared`；`fence`→`sync_work_items`；`async` 循环中的 store→`store_async` | `gpu/phase/outline.cpp` |
| `affine.lower_index`、`cps.conv`、`affine.lower_for` | 仿射索引、直接风格调用、`affine.For` | 核心算术、CPS、递归函数 | `affine`、`cps` 插件 |
| … `ll.emit`、`nvptx.emit` | CPS 世界 | `gpu.split_apply` 把 kernel 分到设备世界；设备侧发射 NVVM IR（内联 PTX），经 `opt`/`llc`/`ptxas`/`fatbinary` 嵌入主机 `.ll` | `nvptx/phase/nvptx.cpp`、`nvptx/nvptx.cpp` |

## 2. 前端到 `group_default`

### 2.1 源程序

调度写在 Mim 里，层级链的类型保证作用域合法（例如 `warp_mma` 只能出现在 `WarpLevels` 里，`wg_mma (64, N, 16)` 只能匹配 `64 × N` 的 warpgroup tile）：

```mim
let levels = serial_group ((16, 16, 0), 1, 0, group_warp ((16, 8, 0), 1, warp_red ((0, 0, 8), 1, 3, warp_mma (16, 8, 8))));

extern fun gemm16 (a b: tensor.Tensor 2 (16, 16) F32): tensor.Tensor 2 (16, 16) F32 =
    let root = map_reduce ((16, 16), (16, 16, 16)) (mac, 0.0:F32) (proj_map @(3, 2) (0, 1))
        (proj_map @(3, 2) (0, 2), proj_map @(3, 2) (2, 1)) (buf a, buf b);
    return (tile_req (root, (), levels, default_control));
```

每个层级的 `sizes` 是对根的 `n` 个循环（这里 `i, j, k`）的切分，`0` 表示这一级不切该维；`tail` 选择尾块封顶（0）或剥离（1）；`promote` 的第 `i` 位表示把输入 `i` 放进共享内存；`pipeline` 是在飞块数。

### 2.2 `tensor.fuse_tensor` 之后：请求原样保留

通用优化把 `map_reduce` 规范化成 `map_reduce_post`，层级链被展开成名义类型的构造子值。`tile_req` 是公理，不会被 PE 展开。

（`tile_mma.mim`，`tensor.fuse_tensor` 之后）

```mim
extern con gemm16 (a: Tensor, b: Tensor, return_1: Cn Tensor) as _29 =
    let _30 = tensor.buf @ (2, ‹2; 16›, math.F (23, 8)) a;
    let _31 = tensor.buf @ (2, ‹2; 16›, math.F (23, 8)) b;
    let root = tensor.map_reduce_post @ (2, 0) @ (math.F (23, 8), math.F (23, 8), 2, 3, SchedT) (‹2; 16›, ‹3; 16›, _1) @ (…) (mac, 0:(math.F (23, 8)), _5) _7 ((_10, _13), ()) ((_30, _31), ());
    let _32 = tensor.tile_req @ (…) (root, (),
        (| _0 | _1: … | _2: [sizes: «3; Nat», Nat, Nat, GroupLevels] | …)#2 ((16, 16, 0), 1, 0,
          (| … | _6: [sizes_3: «3; Nat», Nat, WarpLevels_1] | …)#6 ((16, 8, 0), 1,
            (| _0 | _1: [«3; Nat», Nat, Nat, Nat, WarpLevels_2] | _2: …)#1 ((0, 0, 8), 1, 3, 0,
              (| _0 | _1: … | _2: [«3; Nat», tensor.Tf32Mma, tensor.ScopeTile @ 3 (16, 8, 0)])#2
                ((16, 8, 8), tensor.warp_mma_operands (16, 8, 8), tensor.scope_tile @ 3 (16, 8, 0)) …) …) …) …,
        default_control);
    return_1 _32;
```

`#2`、`#6`、`#1`、`#2` 依次是 `TileLevels#Group`、`GroupLevels#Warp`、`WarpLevels#Red`、`WarpLevels#Mma`。

### 2.3 `tensor.group_default` 之后：调度变成实现原语

`GroupDefault::start`（`group_rules.cpp` 约 492 行）在 `tile-fuse=1` 时运行 `TileFuse`。
日志：`tile_fuse.cpp:1843: tensor.tile_fuse runs root_N as its law`。

定律解释器的做法（`tensor.mim` `mod law` 的 `build`）：

- 每个并行层级（`Par`/`Group`/`Item`/`Warp`/`Warpgroup`）用 `split_par` 生成 `tensor.tiled (ext, ts, tail) xs body`，`body` 被 `wrap` 包上 `tensor.bind scope`：scope 1 = work-group、2 = work-item、3 = warp、6 = warpgroup（`tensor.bind` 的文档在约 2860 行）；`RedWs` 的外层再包 `tensor.specialize pingpong`，`Cluster`/`Persistent` 包 `tensor.cluster`/`tensor.persistent`。
- 每个归约层级用 `split_bound`（`split_red`/`split_pro`）生成 `tensor.tile_fold (count, unit) acc xs step`，每一步体 `body` 视情况包 `tensor.fill_tma`（`RedTma`/`RedWs`）、`tensor.pipeline ring`（`ring > 1`）、`tensor.bind level`（`RedItems`/`Lanes` 这种由 work-item 或 lane 分摊的归约）。
- `promote` 的输入：`plan`/`panel_of` 决定面板的形状与布局，`values_of` 生成 `tensor.slice` → `tensor.stage` → `tensor.placed (cap, layout)`；叶子通过 `panel` 映射从面板读。
- 叶子 `leaf` 是块上的 `map_reduce_post`；`Mma`/`Wgmma` 叶子包 `tensor.mma (n, shape)`。

（`tile_mma.mim`，`tensor.group_default` 之后，节选）

```mim
lam seeded (_10: «2, 2; Nat») (_: Tensor, _: Tensor) as xs2: Tensor_2 =          // 一个 work-group 的 16×16 tile
    …
    lam seeded_1 (_17: «2, 2; Nat») (_: Tensor, _: Tensor) as xs2_1: Tensor_3 =   // 一个 warp 的 16×8 tile
        let acc_1 = tensor.splat @ (math.F (23, 8), 2) (_17#tt, 0:(math.F (23, 8)));
        …
        lam step (a: Tensor_3, k0: Nat) as _32: «2; Tensor» → Tensor_3 =
            lam body (p: Tensor, p_1: Tensor) as xs2_2: Tensor_3 =
                let box = tensor.slice @ (math.F (23, 8), 2) ‹2; 16› ((_10#‹2; ff›, k0), ‹2; 1›, (_10#(tt, ff), 8)) p;
                let stg = tensor.stage @ (math.F (23, 8), 2, 1) (_10#(tt, ff), 8) (0, 16, (tt, ff), ⊥:(math.F (23, 8))) 0 box;
                let put = tensor.placed @ (math.F (23, 8), 3) @ (8, _13, 16) ((8, 1, 16), (…) inj Expr) stg;
                let box_1 = tensor.slice @ (…) ((k0, _10#(ff, tt)), ‹2; 1›, (8, _10#‹2; tt›)) p_1;
                let stg_1 = tensor.stage @ (…) (8, _10#‹2; tt›) (1, 16, (ff, tt), …) 0 box_1;
                let put_1 = tensor.placed @ (…) stg_1;
                let _33 = tensor.map_reduce_post @ (2, 0) @ (…) (_17#tt, (_17#(tt, ff), _17#‹2; tt›, 8), _1) @ (…)
                              (mac, (_17#tt, a), _5) _7 ((panel, panel_1), ()) ((put, put_1), ());
                tensor.mma @ Tensor_3 (3, (16, 8, 8)) _33;
            tensor.pipeline @ («2; Tensor» → Tensor_3) 2 body;
        tensor.tile_fold @ Tensor_3 (2, 8) acc_1 @ «2; Tensor» xs2_1 step;
    tensor.tiled @ (math.F (23, 8), 2) (_10#tt, (16, 8), 1) @ «2; Tensor» xs2 (tensor.bind @ (…) 3 seeded_1);

extern con gemm16 (a_1: Tensor, b: Tensor, return_1: Cn Tensor) as _34 =
    let p_2 = tensor.buf @ (2, ‹2; 16›, math.F (23, 8)) a_1;
    let p_3 = tensor.buf @ (2, ‹2; 16›, math.F (23, 8)) b;
    let _35 = tensor.tiled @ (math.F (23, 8), 2) (‹2; 16›, ‹2; 16›, 1) @ «2; Tensor» (p_2, p_3) (tensor.bind @ (…) 1 seeded);
    return_1 _35;
```

读法：外层 `tiled … (bind 1 seeded)` 是 work-group 层；内层 `tiled … (bind 3 seeded_1)` 是 warp 层；`tile_fold (2, 8)` 把 k 切成 2 块 8；每块把 A、B 的 box 暂存为面板（`stage`+`placed`），叶子在面板上做 `map_reduce_post`，并标记为 `mma (16, 8, 8)`；`pipeline 2` 让下一块的面板与当前块的计算重叠。

## 3. `tensor.lower_to_mem`：分叉发生的地方

`LowerToMem`（`lower_to_mem.cpp`）是一个 `RWPhase`，按 `rewrite_imm_App`（约 682 行）分派：

| 原语 | 函数（约行号） | 产物 |
|---|---|---|
| `tensor.tiled` | `lower_tiled`（1636） | `affine.For` 覆盖所有 tile；`bind` scope 决定循环体包 `gpu.bind.group`/`item`/`warp`/`warpgroup`/`specialized`（约 1804 行）；`persistent` 时外加 `tileRounds` 循环；`specialize` 时插入 `gpu.reg_budget` |
| `tensor.tile_fold` | `lower_tile_fold`（3613） | `tileFold`/`tileFoldStep` 循环；按 `pipeline`/`fill_tma`/`specialize` 生成 ring、`cp.async` 组、TMA + mbarrier、生产者/消费者分支 |
| `tensor.stage` | `lower_stage`（2122） | 共享内存面板分配与填充（协作循环 `bind.cooperative`/`chained`/`async`，或批量阶段令牌） |
| `tensor.placed` | 记入 `placed_layouts_` | 面板的物理布局（来自 `panel_layout.cpp` 的布局搜索） |
| `tensor.mma` + 叶子 `map_reduce_post` | `lower_mma`（2576） | `gpu.buf_ldmatrix` + `gpu.mma_tf32`/`mma_f16`/`mma_bf16`，或 `wgmma_desc`（2566）+ `gpu.wgmma_f16`/`bf16` |
| 其余叶子 `map_reduce_post` | `lower_map_reduce`（1242） | `btensor.map_reduce_post`（留给后面的 `btensor.lower_map_reduce`） |
| 由 work-item 分摊的归约 | `lower_bound_fold`（4301） | 部分和面板 + 阶段令牌 + 合并 |

分叉的依据完全来自 `group_default` 留下的标记：`bind` 的 scope、`tile_fold` 体上的 `pipeline`/`fill_tma`、`tiled` 体上的 `specialize`、叶子上的 `mma`。各路径的细节见 §5–§7。

## 4. `lower_to_mem` 之后的公共阶段

### 4.1 `gpu.check_sync` / `gpu.erase_phases`

同步面板（未流水化、所有 work-item 一起填一起读的面板）在 `lower_to_mem` 里用“工作组阶段”类型表达（`gpu.mim` 的 *Work-group phases* 一节）：`GM (g, e, p)` 是构造 `g` 第 `e` 个超步阶段 `p` 的内存令牌，`SharedBuf (g, e, v, …)` 是带可见性 `v`（`Empty`/`Unpublished`/`Published`/`Owned`）的面板。
`check_sync` 检查类型表达不了的线性使用（令牌恰好用一次、不越出生成它的函数、构造不重入）；`erase_phases`（48 行）把它们擦成普通内存：

- `group_enter`、`sync_publish`、`sync_reclaim` → `gpu.fence`（屏障）；
- `shared_panel` → `buffer.alloc` 于 `gpu.panel_scope`；`coop_fill (m, buf, fill)` → `fill (m, buf)`；
- `enter_write`/`leave_reading`/`view`/`publish_view`/`read_phase` → 直接透传。

例子见 §5.1。

### 4.2 `tensor.distribute`

只对 warp 拥有的区域（`bind 5`/`7`）里的 warp 值起作用：`«32; T»` 的值只保留本 lane 的分量，读别的 lane 变成 `shfl_xor`/`shfl_idx`，`mma_warp` 变成每个 lane 的 `gpu.mma_tf32`。
本文的三个示例都不走这条（它们的 MMA 是 `lower_mma` 直接产生的），所以 `distribute` 在这里是空操作。

### 4.3 `buffer.lower_ptr`

`buffer.Buf (r, s, T)` 变成 `core.Ptr («s; T», addr)`，`buffer.read`/`write` 变成 `mem.lea` + `mem.load`/`store`，`gpu.buf_ldmatrix` 变成 `gpu.ldmatrix`，`gpu.buf_addr` 变成 `mem.lea` + `gpu.shared_addr`。

（`tile_mma.mim`，`buffer.lower_ptr` 之后，面板填充的一个协作循环；`gpu.bind.*` 仍在）

```mim
                let _37 = mem.lea @ (8, ‹8; «5, 4; math.F (23, 8)»›, 0) (_33, _36);
                let _43 = mem.lea @ (5, ‹5; «4; math.F (23, 8)»›, 0) (_37, _42);
                …
                        let _60 = mem.load @ («4; math.F (23, 8)», 0) (_59, _56);
                …
                let _62 = mem.store @ («4; math.F (23, 8)», 0) (_27, _43, _61);
                …
                affine.For @ (0, 2, (core.M 0, core.Ptr («8, 20; math.F (23, 8)», 0))) (gpu.bind.async @ (…) generate) _66 (0I64, 32I64, 1I64, …);
```

### 4.4 `gpu.outline`

`Outline`（`gpu/phase/outline.cpp`）把每个 `gpu.bind.group` 的 `affine.For` 提成一个 kernel：

- `rewrite_imm_App`（约 100 行）：`bind.group` 循环 → `outline`（约 256 行）生成 `con tileKernel (…, _: Idx <groups>, _: Idx <items>, …)`，主机侧变成 `mem.alloc` + `gpu.copy_to_device.block` + `gpu.launch (groups, items, stream, …)` + `copy_to_host`；
- `bind.item`/`warp`/`warpgroup`/`specialized` 循环（约 122–175 行）→ 由线程号算出的迭代号（warp = `tid / 32`，lane = `tid & 31`，warpgroup = `tid >> 7`）；
- `bind.cooperative`/`chained`/`async` 循环（约 194 行）→ 按 work-item 分摊迭代，`cooperative` 前后加 `sync_work_items`，`async` 中的 `mem.store` 到共享内存变成 `gpu.store_async`（`cp.async`）；
- `gpu.shared_scope` 的分配 → `gpu.slot_shared`（静态共享内存槽），`gpu.fence` → `gpu.sync_work_items`。

（`tile_mma.mim`，`gpu.outline` 之后）

```mim
con tileKernel (_6: core.M 1, _7: core.M 3, _8: core.M 4, _9: core.M 5, _: Idx 1, _10: Idx 64, _: [],
                (_11: core.Ptr («16, 16; math.F (23, 8)», 1), _12: …, _13: …) as _14, _15: Cn […]) as _16@(tt) =
    let _17 = core.conv.u @ 64 0 _10;
    let _18 = core.bit2.and 0 @ 0 (31I64, _17);
    let _19 = core.bitcast @ I64 Nat _18;
    let lane = core.idx 0 32 _19;
    …
                let _130 = gpu.store_async @ «4; math.F (23, 8)» (_95, _111, _129);
    …
            let _179 = gpu.async_wait_group 0 _82;
            let _180 = gpu.sync_work_items (gpu.global_mem _179, _7);
    …
                let _295 = gpu.slot_shared «16, 12; math.F (23, 8)» _294;
                let _296 = gpu.slot_shared «8, 20; math.F (23, 8)» _295#ff;
…
fun tiled_1 (…) =
    let _319 = mem.alloc («16, 16; math.F (23, 8)», 1) (gpu.auto_init _312)#1₃;
    let _320 = gpu.copy_to_device.block @ «16, 16; math.F (23, 8)» (…, _315, _319#tt);
    …
    gpu.launch @ (3, ‹3; core.Ptr («16, 16; math.F (23, 8)», 1)›) (1, 64, gpu.default_stream, ff, ()) tileKernel (_319#tt, _321#tt, _323#tt) (…, afterTileKernel);
```

一个 work-group（`Idx 1`），64 个 work-item（两个 warp）。

### 4.5 CPS 化与发射

`affine.lower_index` 把仿射索引代数降成核心算术，`cps.conv` 把直接风格调用（`cps2ds_dep`）变成 CPS，`affine.lower_for` 把 `affine.For` 变成递归函数，之后是闭包转换、类型擦除等通用阶段。
`nvptx.emit`（`nvptx/nvptx.cpp` 的 `Emit::start`）先跑 `gpu.split_apply` 把 kernel 分离到设备世界，`DeviceEmitter`（`nvptx/phase/nvptx.cpp`）把 `gpu.*` 公理发射成 NVVM intrinsic 或内联 PTX（`isa_targetspecific_intrinsic`，约 1008 行起），然后调用 `opt -passes=nvvm-reflect,default<O3>`、`llc -mcpu=sm_XX`、`ptxas`、`fatbinary`，最后 `HostEmitter` 把 fatbin 嵌进主机 `.ll`。
本文没有单独转储 `affine.lower_for` 之后的 IR。

## 5. SIMT 路径

**分叉点**：链里没有 `Warp`/`Warpgroup` 层，只有 `Group` → `Red`/`Item` → `ItemLevels`；叶子是普通 `map_reduce_post`（没有 `mma`）。`group_default` 产生 `bind 1`（work-group）与 `bind 2`（work-item）。

### 5.1 同步面板：`tile_outline_panel.mim`

**`group_default` 之后**：一个 `tile_fold (1, 16)`（k 只有一块），每块把 A 和 B 的 box 暂存为面板，再在面板上按 work-item 平铺（`bind 2`）：

```mim
lam seeded (_9: «2, 2; Nat») (_: Tensor, _: Tensor) as xs2: Tensor_2 =
    let acc_1 = tensor.splat @ (math.F (23, 8), 2) (_9#tt, 0:(math.F (23, 8)));
    …
    lam step (a: Tensor_2, k0: Nat) as _16 (_: Tensor, _: Tensor) as xs2_1: Tensor_2 =
        lam carried (_17: «2, 2; Nat») ((p: Tensor, p_1: Tensor) as xs2_2, acc2: Tensor_2) as ins: Tensor_4 =
            …
            let box = tensor.slice @ (math.F (23, 8), 2) ‹2; 16› ((_9#‹2; ff›, k0), ‹2; 1›, (_9#(tt, ff), 16)) p;
            let stg = tensor.stage @ (math.F (23, 8), 2, 1) (_9#(tt, ff), 16) (0, 4, (tt, ff), ⊥:(math.F (23, 8))) 0 box;
            let put = tensor.placed @ (math.F (23, 8), 3) @ (16, _12, 4) ((16, 2, 4), (…) inj Expr) stg;
            …
            tensor.map_reduce_post @ (2, 0) @ (…) (_17#tt, (_17#(tt, ff), _17#‹2; tt›, 16), _1) @ (…) (mac, (_17#tt, acc_2), _5) out ((panel, panel_1), ()) ((put, put_1), ());
        tensor.tiled @ (math.F (23, 8), 2) (_9#tt, ‹2; 4›, 0) @ [«2; Tensor», Tensor_2] (xs2_1, a) (tensor.bind @ (…) 2 carried);
    tensor.tile_fold @ Tensor_2 (1, 16) acc_1 @ «2; Tensor» xs2 step;

extern con gemm16 (…) =
    …
    let _33 = tensor.tiled @ (math.F (23, 8), 2) (‹2; 16›, ‹2; 8›, 0) @ «2; Tensor» (p_2, p_3) (tensor.bind @ (…) 1 seeded);
```

注意面板布局 `(16, 2, 4)`：块深 16 > 8 时布局搜索把 k 放到最内层（测试注释所述）。

**`lower_to_mem` 之后**：面板加入它的消费者所在的“批”，用工作组阶段令牌表达一个超步——进入、协作填充、发布、读、离开（`lower_stage` 约 1030–1060 行的批量分支）：

```mim
                con tileItems (_159: I64, (_160: core.M 0, _161: buffer.Buf (2, ‹2; 8›, math.F (23, 8))) as _162, _163: Cn […]) as _164@(tt) =
                    let _165 = gpu.group_enter @ 0 _160;
                    let _166 = gpu.enter_write @ ‹2; 0› _165;
                    let _167 = gpu.shared_panel @ ‹2; 0› @ (3, (2, 4, 16), math.F (23, 8)) _166;
                    let _168 = gpu.coop_fill @ ‹2; 0› @ (3, (2, 4, 16), math.F (23, 8)) (_167#ff, _167#tt, panelFill);
                    let _169 = gpu.shared_panel @ ‹2; 0› @ (3, (16, 2, 4), math.F (23, 8)) _168#ff;
                    let _170 = gpu.coop_fill @ ‹2; 0› @ (3, (16, 2, 4), math.F (23, 8)) (_169#ff, _169#tt, panelFill_1);
                    let _171 = gpu.sync_publish @ ‹2; 0› _170#ff;
                    …
                    let _204 = gpu.publish_view @ ‹2; 0› @ (3, (2, 4, 16), math.F (23, 8)) (_171, _168#tt);
                    let _205 = gpu.view @ ‹2; 0› @ (3, (2, 4, 16), math.F (23, 8)) _204;
                    …
                    lam batchRead (_208: core.M 0): [core.M 0, buffer.Buf (2, ‹2; 8›, math.F (23, 8))] =
                        btensor.map_reduce_post @ (2, 0) @ (…) (‹2; 8›, (4, 4, 16), _2) @ (…) …
                    let _209 = gpu.read_phase @ ‹2; 0› @ (buffer.Buf (2, ‹2; 8›, math.F (23, 8))) (_171, batchRead);
                    let _210 = gpu.leave_reading @ ‹2; 0› _209#ff;
                    _163 (_210, _209#tt);
                …
                affine.For @ (0, 2, (…)) (gpu.bind.item @ (…) tileItems) tileScratchDone (0I64, 4I64, 1I64, …);
        …
        affine.For @ (0, 2, (…)) (gpu.bind.group @ (…) tileGroups) tileScratchDone_1 (0I64, 4I64, 1I64, …);
```

面板填充体 `panelFill` 里是 `gpu.bind.chained` 的协作循环（无屏障，由随后的 `sync_publish` 发布）。

**`erase_phases` 之后**：令牌被擦掉，同一处变成

```mim
                    let _191 = gpu.panel_scope @ 0 (gpu.fence _154);
                    …
                    let _194 = gpu.panel_scope @ 0 _193#ff;
                    …
                    let _197 = btensor.map_reduce_post @ (2, 0) @ (…) (‹2; 8›, (4, 4, 16), _1) @ (…) …
```

**`gpu.outline` 之后**：`gpu.launch (4, 4, …)`（4 个 work-group × 4 个 work-item），`fence` 变成 `gpu.sync_work_items`，面板变成 `gpu.slot_shared`：

```mim
            let _381 = gpu.sync_work_items (gpu.global_mem _283, _7);
            let _383 = gpu.slot_shared «2, 4, 16; math.F (23, 8)» _382;
            let _386 = gpu.slot_shared «16, 2, 4; math.F (23, 8)» _385;
            let _388 = gpu.sync_work_items (gpu.global_mem _387#ff, _7);
…
    gpu.launch @ (3, ‹3; core.Ptr («16, 16; math.F (23, 8)», 1)›) (4, 4, gpu.default_stream, ff, ()) tileKernel (…) (…);
```

**PTX**（`-X nvptx:sm=90a`，`tile_outline_panel_dev.ptx`）：静态共享内存槽、向量化全局读 + 共享写、屏障、逐元素 `fma`：

```ptx
.visible .entry tileKernel_9492265(
	.param .align 8 .b8 tileKernel_9492265_param_0[24]
)
	.shared .align 16 .b8 _9500852_$_slot[256];
	.shared .align 16 .b8 _9501215_$_slot[512];
	.shared .align 16 .b8 _9501235_$_slot[512];
	…
	bar.sync 	0;
	ld.global.v4.b32 	{%r4, %r5, %r6, %r7}, [%rd17];
	…
	st.shared.v4.b32 	[%rd19], {%r4, %r5, %r6, %r7};
	…
	bar.sync 	0;
	ld.shared.b32 	%r109, [%rd5];
	ld.shared.b32 	%r108, [%rd5+4];
	…
	fma.rn.f32 	%r109, %r70, %r69, %r109;
	fma.rn.f32 	%r108, %r71, %r69, %r108;
```

### 5.2 cp.async 环：`tile_pipeline.mim`

`item_red_pipelined (…, 3, …)` 的 `pipeline 3` 在 `group_default` 之后是 `tensor.pipeline 3`：

```mim
            tensor.pipeline @ («2; Tensor» → Tensor_3) 3 body;
        tensor.tile_fold @ Tensor_3 ‹2; 4› acc_1 @ «2; Tensor» xs2_1 step;
    tensor.tiled @ (math.F (23, 8), 2) (_9#tt, ‹2; 2›, 1) @ «2; Tensor» xs2 (tensor.bind @ (…) 2 seeded_1);
…
    let _33 = tensor.tiled @ (math.F (23, 8), 2) (‹2; 16›, ‹2; 8›, 1) @ «2; Tensor» (p_2, p_3) (tensor.bind @ (…) 1 seeded);
```

`lower_tile_fold` 为每个面板分配 3 个缓冲（ring），面板填充循环标成 `gpu.bind.async`（`lower_stage` 约 534 行，`async_fill_` 时），在主循环每一步：`async_wait_group` 等最早的组、`fence`、对 `n − 1` 块之后的块发起填充、`async_commit`，然后计算当前块（MMA 例子的 §6.2 片段结构相同）。
`gpu.outline` 把 `async` 循环里的共享内存写换成 `gpu.store_async`，`nvptx` 把它发射成 `cp.async`：

```ptx
	.shared .align 16 .b8 _9468429_$_slot[128];
	… （共 6 个 128 字节槽：2 个输入 × 3 级 ring）
	cp.async.cg.shared.global [%rd9], [%rd6], 16;
	cp.async.ca.shared.global [%rd15], [%rd14], 4;
	cp.async.ca.shared.global [%rd15+64], [%rd14+128], 4;
	cp.async.commit_group;
	…
	cp.async.commit_group;
	cp.async.wait_group 	1;
	bar.sync 	0;
	cp.async.cg.shared.global [%rd23], [%rd22], 16;
	…
	cp.async.commit_group;
	ld.shared.b32 	%r2, [%rd26];
```

`wait_group 1` 对应 ring 深度 3：始终有 1 组在飞。

## 6. MMA 路径（Ampere `mma.sync`）

**分叉点**：`GroupLevels#Warp`（`group_warp`）进入 `WarpLevels`，叶子是 `WarpLevels#Mma`（`warp_mma (16, 8, 8)`，类型要求根的收缩种类是 `Tf32Mma`，F16/BF16 输入用 `(16, 8, 16)` 与 `HalfMma`）。
`group_default` 产生 `bind 3` 与 `tensor.mma (3, (16, 8, 8))`（§2.3 的片段）。

### 6.1 `lower_to_mem`

`lower_tiled` 把 `bind 3` 的循环标成 `gpu.bind.warp`；`lower_mma` 把带 `mma` 标记的叶子降成 tensor-core 指令：A 的片段从面板用 `gpu.buf_ldmatrix` 取（约 2688 行），B 的片段按 lane 读，累加器是每个 lane 的 `«4; F32»`，最后 `gpu.mma_tf32`（约 2710 行；F16/BF16 是 `mma_f16`/`mma_bf16`）。`pipeline 2` 使面板填充是 `gpu.bind.async` 循环、配 `async_commit`/`async_wait_group`。

（`tile_mma.mim`，`lower_to_mem` 之后，一步 k 块）

```mim
                con tileFoldStep (_125: I64, (_126: core.M 0, _127: «4; math.F (23, 8)», _128: buffer.Buf (2, (16, 12), math.F (23, 8)),
                                  _129: buffer.Buf (2, (8, 20), math.F (23, 8)), _130: …, _131: …) as _132, _133: Cn […]) =
                    let iteration = core.bitcast @ I64 Nat _125;
                    …
                    con generate_7 (…) = affine.For @ (…) (gpu.bind.async @ (…) generate_6) _213 (0I64, 32I64, 1I64, (_210, _211));
                    let _215 = gpu.async_wait_group 0 _126;
                    let _216 = cps.cps2ds_dep @ […] Uf_6 generate_7 (gpu.fence _215, _130, ());     // 填下一块的 A 面板
                    let _217 = cps.cps2ds_dep @ […] Uf_4 generate_5 (_216#ff, _131, ());            // 填下一块的 B 面板
                    let _218 = affine.map @ ‹2; 3› @ ((16, 8, 8), (1, 16, 12)) placedMap (… gpu.lane_id () …);
                    let _219 = buffer.vectorize (2, (16, 12), math.F (23, 8)) (4, (16, 3)) _128;
                    …
                    let _223 = gpu.buf_ldmatrix @ (2, (16, 3)) (4, _218#ff, _219, (_218#tt#1₃, _222));
                    let _225 = buffer.read (2, (8, 20), math.F (23, 8)) (_224#ff, _129, (_224#tt#0₃, _224#tt#2₃));
                    let _227 = buffer.read (2, (8, 20), math.F (23, 8)) (_226#ff, _129, (_226#tt#0₃, _226#tt#2₃));
                    let _228 = gpu.mma_tf32 (_223#tt, (_225#tt, _227#tt), _127);
                    _133 (_227#ff, _228, _130, _131, _128, _129);          // 交换 ring 的两个槽
                …
                    affine.For @ (0, 6, (…)) tileFoldStep tileFoldDone (0I64, 2I64, 1I64, _229);
                …
                        let _238 = buffer.alloc (2, (16, 12), math.F (23, 8)) (gpu.shared_scope _237);
                        …
                        let _246 = cps.cps2ds_dep @ […] Uf_3 tileFold (gpu.async_commit _245#ff, …);
                        let _247 = buffer.write (2, (16, 8), «2; math.F (23, 8)») (_100, _105, (_15, _112), (_246#tt#0₄, _246#tt#1₄));
                        let _248 = buffer.write (2, (16, 8), «2; math.F (23, 8)») (_247#ff, _105, (_18, _112), (_246#tt#2₄, _246#tt#3₄));
            …
            affine.For @ (0, 2, (…)) (gpu.bind.warp @ (…) tileWarps) tileScratchDone (0I64, 2I64, 1I64, _19);
    …
    affine.For @ (0, 2, (…)) (gpu.bind.group @ (…) tileGroups) tileScratchDone_1 (0I64, 1I64, 1I64, _6);
```

面板形状 `(16, 12)`、`(8, 20)` 是带填充的布局（避免 bank 冲突），由 `placed` 的布局决定。
累加器的 4 个分量按 `mma.m16n8k8` 的 C 布局写回（两次 `buffer.write`，行 `lane/4` 与 `lane/4 + 8`）。

### 6.2 `gpu.outline` 与 PTX

outline 之后：`gpu.ldmatrix`、`gpu.mma_tf32`、`gpu.store_async`、`gpu.async_wait_group 0` + `gpu.sync_work_items`，`gpu.launch (1, 64, …)`（见 §4.4）。

`tile_mma_dev.ptx`：

```ptx
	.shared .align 16 .b8 _10114206_$_slot[768];
	.shared .align 16 .b8 _10114208_$_slot[768];
	.shared .align 16 .b8 _10114217_$_slot[640];
	.shared .align 16 .b8 _10114241_$_slot[640];
	…
	cp.async.commit_group;
	cp.async.wait_group 	0;
	bar.sync 	0;
	…
	cp.async.cg.shared.global [%rd33], [%rd11+32], 16;
	cp.async.cg.shared.global [%rd35], [%rd14+512], 16;
	cp.async.commit_group;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10, %r11, %r12, %r13}, [%rd38];
	ld.shared.b32 	%r14, [%rd41];
	ld.shared.b32 	%r15, [%rd41+320];
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%r2, %r3, %r4, %r5},
		{%r10, %r11, %r12, %r13},
		{%r14, %r15},
		{%r16, %r16, %r16, %r16};
	cp.async.wait_group 	0;
	bar.sync 	0;
```

TF32 的 A 片段借用 `ldmatrix.b16`（每个 32 位元素是两个 b16）；面板 768 = 16×12×4 字节，640 = 8×20×4 字节，各两份（ring 深度 2）。

## 7. Hopper wgmma + warp specialization 路径

**分叉点**：`GroupLevels#Warpgroup`（`group_warpgroup ((64, 128, 0), …)`）进入 `WarpgroupLevels`，其类型以 work-group tile 与 warpgroup tile 为索引；叶子是 `Wgmma`（`wg_mma (64, 128, 16)`，只接受 `HalfMma`，即 F16/BF16 输入）。
归约层有三种：`Red`（普通流水，`wg_red`/`wg_red_pipelined`）、`RedTma`（`wg_red_tma`：领头线程用 TMA 填 ring，每槽一个 mbarrier）、`RedWs`（`wg_red_ws`：多出一个生产者 warp 只做 TMA，消费者 warpgroup 只算；`pingpong` 1 时两个消费者 warpgroup 轮流发 MMA）。

### 7.1 `group_default` 之后

（`tile_wgmma_ws.mim`）

```mim
        lam step (a: Tensor_3, k0: Nat) as _34: «2; Tensor» → Tensor_3 =
            lam body (p: Tensor, p_1: Tensor) as xs2_2: Tensor_3 =
                let box = tensor.slice @ (math.F (10, 5), 2) ‹2; 128› ((_12#‹2; ff›, k0), ‹2; 1›, (_12#(tt, ff), 64)) p;
                let stg = tensor.stage @ (math.F (10, 5), 2, 1) (_12#(tt, ff), 64) (0, 128, (tt, ff), ⊥:(math.F (10, 5))) 0 box;
                let put = tensor.placed @ (math.F (10, 5), 3) @ (64, _15, 128) ((64, 1, 128), ((‹(64, 1, 128)›, 1, (2₄, 1₄, 3₄, 0₄), ‹3; 0›, 8, 8, 0, 8, tt) inj Data) inj Expr) stg;
                …
                let _35 = tensor.map_reduce_post @ (2, 0) @ (…) (_19#tt, (_19#(tt, ff), _19#‹2; tt›, 64), _1) @ (‹2; math.F (10, 5)›, …) (mac, (_19#tt, a), _7) _9 ((panel, panel_1), ()) ((put, put_1), …);
                tensor.mma @ Tensor_3 (3, (64, 128, 16)) _35;
            tensor.pipeline @ («2; Tensor» → Tensor_3) 2 (tensor.fill_tma @ («2; Tensor» → Tensor_3) body);
        tensor.tile_fold @ Tensor_3 (2, 64) acc_1 @ «2; Tensor» xs2_1 step;
    tensor.tiled @ (math.F (23, 8), 2) (_12#tt, (64, 128), 1) @ «2; Tensor» xs2
        (tensor.bind @ (…) 6 (tensor.specialize @ (…) 0 seeded_1));
…
    let _37 = tensor.tiled @ (math.F (23, 8), 2) (‹2; 128›, ‹2; 128›, 1) @ «2; Tensor» (p_2, p_3) (tensor.bind @ (…) 1 seeded);
```

与 MMA 路径的差别：`bind 6`（warpgroup）、`specialize 0`（WS、非 ping-pong）、`fill_tma`、`mma (64, 128, 16)`；面板布局的最后一个字段 `tt` 与 `8, 8, 0, 8` 表示 128 字节 swizzle 的 K-major 原子布局，供 wgmma 描述符与 TMA 共用。
`tile_wgmma_tma.mim`（`RedTma`）在这一步只少了 `specialize`：`tensor.bind @ (…) 6 seeded_1`，其余相同。

### 7.2 `lower_to_mem` 之后

- `lower_tiled`：`bind 6` + `specialize` → `gpu.bind.specialized`（最后一次迭代只跑一个 warp），并在 tile 状态上挂 `gpu.reg_budget 168`（约 1866 行：`16384 / (32 × 每分区 warp 数)` 向下取 8 的倍数）。
- 主机侧为两个输入各建一个 TMA 张量图 `gpu.buf_tensor_map`（约 3867 行；A 的盒 128×64，B 的盒 64×64）。
- `lower_tile_fold`（约 3860–3960 行）：ring 的每个槽一个 “full” mbarrier（`count 1`，生产者 `expect_tx` 字节数）和一个 “empty” mbarrier（`count 8` = 两个消费者 warpgroup 的 8 个 warp）；每一步按角色分支 `(foldConsume, foldProduce)#is_producer`。
- `lower_mma` 的 wgmma 分支（约 2642 行）：每个 k 子块（64/16 = 4 个）用 `wgmma_desc` 从面板地址算 64 位描述符，发一条 `gpu.wgmma_f16 (16, …)`（`nt = 128/8 = 16`），前加 `wgmma_fence`，后 `wgmma_commit` + `wgmma_wait`。

```mim
    let _77 = gpu.buf_tensor_map @ (math.F (10, 5)) @ ‹2; 128› (a, 128, 64);
    let _78 = gpu.buf_tensor_map @ (math.F (10, 5)) @ ‹2; 128› (b, 64, 64);
…
            con tileWarpgroups (_139: I64, (_140: core.M 0, _141: buffer.Buf (2, ‹2; 128›, math.F (23, 8))) as _142, _143: Cn […]) as _144@(tt) =
                …
                let _155 = core.ncmp.glE (2, _146);                         // 第 2 个迭代（生产者）
                …
                        con tileFoldStep (_289: I64, (_290: core.M 0, _291: «16, 4; math.F (23, 8)», _292: buffer.Buf (2, (128, 64), math.F (10, 5)),
                                          _293: buffer.Buf (3, (2, 64, 64), math.F (10, 5)), …) as …) =
                            let _302 = gpu.buf_addr @ I64 @ (1, 2) (_283#tt, _301);   // full[slot]
                            let _306 = gpu.buf_addr @ I64 @ (1, 2) (_288#tt, _301);   // empty[slot]
                            con foldConsume (_307: core.M 0)@(tt) =
                                let _308 = gpu.mbar_wait_at (_307, _302, _305);
                                let addr = gpu.buf_addr @ (math.F (10, 5)) @ (2, (128, 64)) (_292, (…));
                                let addr_1 = gpu.buf_addr @ (math.F (10, 5)) @ (3, (2, 64, 64)) (_293, _315#tt);
                                let _319 = gpu.wgmma_f16 (16, gpu.wgmma_fence _310#ff, _314, _318, tt, _291);
                                …
                                let _346 = gpu.wgmma_f16 (16, _337#ff, _341, _345, tt, _337#tt);
                                let _347 = gpu.wgmma_wait @ «16, 4; math.F (23, 8)» (gpu.wgmma_commit _346#ff, 0, _346#tt);
                                …                                                    // mbar_arrive_if 到 empty[slot]
                                _297 (_348, _347#tt, _294, _295, _292, _293);
                            con foldProduce (_357: core.M 0)@(tt) =
                                let _358 = gpu.mbar_wait_if (_357, _306, _351, _279);
                                let _359 = gpu.mbar_expect_tx_if (_358, _302, 32768I32, _279);
                                let _360 = gpu.tma_load_if (_359, _352, _77, _354, _136, _302, _279);
                                let _361 = gpu.tma_load_if (_360, _355, _78, _137, _354, _302, _279);
                                let _362 = gpu.tma_load_if (_361, _356, _78, _138, _354, _302, _279);
                                _297 (_362, _291, _294, _295, _292, _293);
                            (foldConsume, foldProduce)#_155 _290;
                        …
                        let _369 = gpu.mbar_init_if (_288#ff, _368, 8I32, gpu.leader ());
                        let _371 = gpu.mbar_init_if (_369, _370, 8I32, gpu.leader ());
                …
                affine.For @ (0, 2, (…)) (gpu.bind.specialized @ (…) …) tileScratchDone (…);
        …
        affine.For @ (0, 2, (…)) (gpu.bind.group @ (…) tileGroups) tileScratchDone_1 (…);
```

`32768` = A 盒 128×64×2 + B 两个盒 64×64×2 字节；B 用两次 TMA（列偏移 0 与 64）填满 128 列。
`(foldConsume, foldProduce)#_155` 是无分支的角色选择，`gpu.*_if` 的 `issue` 谓词让只有一个线程实际发 TMA/expect_tx。

### 7.3 `gpu.outline` 之后

288 个 work-item = 2 个消费者 warpgroup（256）+ 1 个生产者 warp（32）；`bind.specialized` 的迭代号由 `tid >> 7` 得到；`gpu.buf_addr` 已成 `gpu.shared_addr`，描述符按位拼出：

```mim
con tileKernel (_76: core.M 1, _77: core.M 3, _78: core.M 4, _79: core.M 5, _: Idx 1, _80: Idx 288, _: [],
                (_81: core.Ptr («16; I64», 1), _82: core.Ptr («16; I64», 1), _83: core.Ptr («128, 128; math.F (10, 5)», 1), _84: …) …) =
    let _89 = core.conv.u @ 288 0 _80;
    let _90 = core.shr.l @ 0 (_89, 7I64);
    …
    let _164 = core.ncmp.glE (2, _91);
    …
    let _245 = core.icmp.xyglE @ 0 (0I64, _89);                     // tid == 0：初始化 mbarrier
    let _246 = core.icmp.xyglE @ 0 (0I64, _95);                     // lane == 0：每个消费者 warp arrive 一次
    let _247 = core.bit2.and 0 @ 0 (127I64, _89);
    let _248 = core.icmp.xyglE @ 0 (0I64, _247);
    let _249 = core.bit2.and 2 @ 2 (_164, _248);                     // 生产者的第一个线程发 TMA
    …
        let _329 = gpu.reg_budget 168 _250;
                let _333 = gpu.slot_shared «128, 64; math.F (10, 5)» _332;
                let _334 = gpu.slot_shared «2, 64, 64; math.F (10, 5)» _333#ff;
                let _335 = gpu.slot_shared «2; I64» _334#ff;
                …
                        let addr = gpu.shared_addr @ (math.F (10, 5)) @ 0 _369;
                        let x = core.conv.u @ i32 0 addr;
                        let _370 = core.shr.l @ 0 (x, 4I64);
                        let _371 = core.bit2.and 0 @ 0 (16383I64, _370);
                        let _372 = core.bit2.or 0 @ 0 (4611686293305360384I64, _371);    // wgmma 描述符
                        …
                        let _380 = gpu.wgmma_f16 (16, gpu.wgmma_fence _366#ff, _372, _379, tt, _345);
…
        let _484 = gpu.device_tensor_map @ (math.F (10, 5)) @ «128, 128; math.F (10, 5)» @ 1 (_483#tt, 128, 128, 128, 64);
        let _487 = gpu.device_tensor_map @ (math.F (10, 5)) @ «128, 128; math.F (10, 5)» @ 1 (_486#tt, 128, 128, 64, 64);
        …
        gpu.launch @ (5, (core.Ptr («16; I64», 1), core.Ptr («16; I64», 1), …)) (1, 288, gpu.…
```

### 7.4 PTX

`tile_wgmma_ws_dev.ptx`（`-X nvptx:sm=90a`）：

```ptx
.extern .shared .align 1024 .b8 tileKernel_9491113_dynamic_shared[];
.visible .entry tileKernel_9491113(
.maxnreg 168
	{ .reg .pred q; setp.ne.b32 q, %r6, 0; @q mbarrier.init.shared::cta.b64 [%r4], %r5; }
	…
	bar.sync 	0;
	…
// %bb.1:                               // %foldProduce_9499429
	{ .reg .pred p, q; setp.ne.b32 q, %r3, 0; @!q bra D0; W0: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%r15], %r5; @!p bra W0; D0: }
	{ .reg .pred q; setp.ne.b32 q, %r3, 0; @q mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r19], %r16; }
	{ .reg .pred q; setp.ne.b32 q, %r3, 0; @q cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r17], [%rd4, {%r18, %r18}], [%r…
	…
	{ .reg .pred p; W1: mbarrier.try_wait.parity.shared::cta.b64 p, [%r12], %r37; @!p bra W1; }
	wgmma.fence.sync.aligned;
	wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r37, %r38, …
	… （每个 k 块 4 条）
	wgmma.commit_group.sync.aligned;
	wgmma.wait_group.sync.aligned 	0;
	…
	{ .reg .pred q; setp.ne.b32 q, %r2, 0; @q mbarrier.arrive.shared::cta.b64 _, [%r13]; }
```

`.maxnreg 168` 来自 `gpu.reg_budget`；面板放在 1024 字节对齐的动态共享内存里（TMA 目标要求 1 KiB 对齐）。

**`setmaxnreg`**：`gpu.mim` 有 `gpu.setmaxnreg` 公理，`nvptx` 也能发射 `setmaxnreg.{inc,dec}.sync.aligned.u32`（`nvptx/phase/nvptx.cpp` 约 1212 行），但当前 `lower_to_mem.cpp` 不产生它；这个示例的 PTX 里没有 `setmaxnreg`，寄存器预算只通过 `.maxnreg` 给出。

**ping-pong**：`wg_red_ws (…, 2, 1, …)`（`tile_wgmma_pingpong.mim`）在 `group_default` 之后是 `specialize 1`，两个消费者 warpgroup 用命名屏障轮流发 MMA：`lower_mma` 在一次 wgmma 序列前 `gpu.sync_named` 自己的屏障、之后 `gpu.arrive_named` 对方的屏障（约 2645–2657 行），`lower_tile_fold` 在开头让第二个 warpgroup 先 `arrive_named_if` 一次（约 3973 行）。本文没有转储这个示例。

**`RedTma`（无特化）**：没有生产者 warp，`bind 6` 变成 `gpu.bind.warpgroup`，由 work-group 的领头线程在同一条指令流里发 TMA，每个 ring 槽一个 mbarrier；本文只验证了它在 `group_default` 之后与 WS 的差别（§7.1）。

## 8. 融合在 IR 中的样子

`law.fuse slot …` 把一个生产者 `map_reduce` 融进消费者定律的第 `slot` 个输入；`law.fuse_shared` 让多个消费者共享生产者，`law.fuse_law` 让一个已融合的定律再作为另一个定律的来源（嵌套融合，带驻留成员）。
定律解释器里，被融合的输入的 `Src` 记录带 `fused = tt`，`values_of` 在填面板时不再切源张量，而是对该块的 box 现算生产者。
`TileFuse::run_as_law` 也是这样处理只有一个成员、被根直接读的 `tile_req`：构造 `law.fuse slot (…生产者…) (…根…)`。

`law_fuse.mim` 的 `simt_run`（`law.run (gemm (a, w, s), simt, …)`，`gemm` 把反量化 `dequant` 融进 GEMM 的 B 输入）。
因为 `law.run` 是普通 Mim 函数，在 `optimize` 阶段就被 PE 展开，`tensor.fuse_tensor` 之后已经是实现原语（`group_default` 之后不变）：

```mim
            let _53 = tensor.slice @ (math.F (23, 8), 2) ‹2; 16› ((_17#‹2; ff›, k0), ‹2; 1›, (_17#(tt, ff), 4)) p;
            let stg = tensor.stage @ (math.F (23, 8), 2, 1) (_17#(tt, ff), 4) (0, 2, (tt, ff), ⊥:(math.F (23, 8))) 0 _53;
            let put = tensor.placed @ (math.F (23, 8), 3) @ (4, f, 2) ((4, 4, 2), (…) inj Expr) stg;
            let box = tensor.map_reduce_post @ (2, 0) @ (math.F (23, 8), math.F (23, 8), 2, 2, SchedT) ((4, _17#‹2; tt›), (4, _17#‹2; tt›), _10)
                          @ ((I8, math.F (23, 8)), ‹2; 2›, (‹2; 16›, (1, 16)), (), (), ()) @ […] (dequant, ((4, _17#‹2; tt›), acc_2), _14) _16 …
            let stg_1 = tensor.stage @ (math.F (23, 8), 2, 1) (4, _17#‹2; tt›) (1, 2, (ff, tt), ⊥:(math.F (23, 8))) 0 box;
            let put_1 = tensor.placed @ (math.F (23, 8), 3) @ (4, f_1, 2) ((4, 4, 2), (…) inj Expr) stg_1;
```

A 的面板是源张量的切片；B 的面板由 `dequant` 的 `map_reduce_post`（输入 `I8` 权重与 `F32` 尺度）对该块的 box 现算，反量化后的权重从不整体物化。
之后的阶段与 §5 相同。

## 9. 不确定与未验证之处

- 行号是 2026-11-16 的近似值；`tensor.mim` 在写作期间多次被修改（转储时偶尔遇到它暂时不能解析，重试后才成功），以函数名为准。
- `affine.lower_index`、`cps.conv`、`affine.lower_for`、闭包转换、`gpu.split_apply` 之后的中间 IR 没有转储，§4.5 的描述来自源码与插件文档。
- `tensor.distribute` 在三个示例里都是空操作，§4.2 的描述来自 `tensor.mim`/`distribute.cpp` 的文档。
- ping-pong（§7.4）和 `RedTma` 的 `lower_to_mem`/PTX 没有转储；ping-pong 使用命名屏障的说法来自 `lower_to_mem.cpp` 源码，未在输出中核对。
- `Cluster`、`Persistent`、`RedItems`、`Lanes` 等层级和 attention 路径不在本文范围内。
- 本文没有在 GPU 上运行任何 kernel；PTX 是 `llc`/`ptxas -arch=sm_90a` 成功编译的结果，但正确性未在本文验证（由对应的 lit 测试负责）。
