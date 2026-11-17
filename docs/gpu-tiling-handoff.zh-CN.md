# GPU tiling：换机器继续开发的交接说明（2026-11-16）

分支 `feat/gpu-tiling-refactor`，交接时的头是 `bcc8c9a7de` 之后的这次提交。
设计与结果的正文在 `docs/tile-law-design.zh-CN.md`（§7 定律解释器，§8 类型索引与 Mim 里的协议）和 `docs/gpu-tiling-design.zh-CN.md`；本文只写接手时需要、别处没有的东西。

## 1. 环境

- 用 clang 构建（GCC 15.2 在 `src/mim/pattern/graph.cpp` 上 ICE）：
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=ON`，然后 `cmake --build build -j$(nproc)`。
- 原机器是 clang/LLVM 22.1.3、cmake 4.3；`nvptx` 的 lit 测试需要 `llc`、`opt`、`llvm-link`、`FileCheck` 和 CUDA（`-lcuda`、libdevice）。
- 提交未签名（原机器上 GPG 密钥未解锁，用了 `--no-gpg-sign`）。
- 仓库里有不属于这条线的未跟踪 WIP 文件（`bench/gpu_fusion/attn_flash.mim`、`var_*.mim`、`docs/gpu-sync-typed-design-agy.zh-CN.md`），所以 `git add` 只写显式路径，不 add 目录。
- 不要用 `git checkout -- <file>` 撤销改动，这个 worktree 上因此丢过未提交的内容。

## 2. 验证方式

- 完整 lit：`cmake --build build --target lit`。
  在 sm_75 的卡（GTX 1650 SUPER）上有 12 个 `nvptx` 测试固定失败，它们需要 sm_80 的 TF32/F16 mma：
  `warp_owned`、`warp_lanes`、`group_tile_req_mma`、`group_tile_req_acc`、`tile_dequant`、`tile_dequant_chain`、`tile_dequant_nested`、`law_blocks`、`law_blocks_roll`、`mma_k16`、`attention_fold`、`tile_mma16`。
  最近一次完整结果是 981 通过、这 12 个失败；在 sm_80 以上的卡上它们应当通过，没有验证过。
- 改编译器内部而不该改变输出时，用 `bench/typed_tiles/tools/`（都从仓库根目录运行）：
  - `snap.sh <dir>`：把 `lit/gpu/*.mim` 全部编译成 Mim 并存下去掉 gid 的输出，改动前后各跑一次再 `diff` 两个 `SUMMARY`。
  - `strip.py <file>`：与名字无关的结构哈希，用在名字或插件顺序变了但结构应当相同的时候。
  - `devsnap.sh <dir>`：wgmma 测试的设备 IR（sm_90a，只编译不执行），比较行数、`mbarrier.try_wait` 与 `cp.async.bulk` 的个数。
  - `ctime.py <基线 worktree>`：12 个 `lit/gpu` 程序的编译时间对比。
  - `mklit.py`：从 `bench/typed_tiles/` 的正例和 `reject_n*.mim` 重新生成 `lit/gpu/typed_tiles.mim`、`lit/nvptx/typed_tiles.mim`、`lit/error/typed_tile_*.mim`。
    改了外壳实验后要重新运行它，再构建 `lit` 目标。
- GPU 上跑单个融合程序：`bench/gpu_fusion/ptx_bench.py prepare|run`。

## 3. 现状

| 部分 | 位置 | 状态 |
|---|---|---|
| 定律解释器的类型索引（执行单元 `At`/`Step u`、累加器 `Acc`、面板计划 `Plan`） | `src/mim/plug/tensor/tensor.mim` 的 `mod law` | 已接入；88 个 `lit/gpu` 输出逐字不变，编译时间无可测变化 |
| 协议骨架：`superstep`、`ring_fold`（`async_ring`、`mbar_ring`、`shared_ring`、`cluster_ring`）、`merge_rounds`、`lane_merge` | `src/mim/plug/gpu/gpu.mim`，由 `lower_to_mem.cpp` 调用 | 已搬到 Mim；TMA 与 cluster 的环只有 sm_90a 设备 IR 与 PTX 的结构对比，没有硬件执行 |
| 类型化数据 tile 外壳：`Tile r unit at life space cap r2 s lay T`、`Ctx` | `bench/typed_tiles/prelude.mim`，lit 在 `lit/gpu/typed_tiles.mim`、`lit/error/typed_tile_*.mim` | 1 个编译检查、1 个 GPU 执行、9 个反例；编译器的解释器没有用这个类型 |
| 缓冲区层解释器 `run_buffers` | `bench/typed_tiles/buffer_interp.mim` | 原型，决定停在这里：不并入编译器，也不追剩下的差距 |
| 仍在 C++ 的协议 | `lower_to_mem.cpp` | 批之外的协作填充、`lower_tiled`、`gpu.outline`；warp 特化、ping-pong、persistent 的环已是 `gpu.ws_fold` 与 `RingSync`，并在 H200 上执行过 |

性能数字（3090 上大方阵 72–85% cuBLAS，22 例几何平均 52%；1650S 上 87%）和 split-K 的结果在基准仓库的 `results/gpu-tiling-*-20261116/comparison.md` 与 `NOTES.md`，见 §5。

## 4. 待办（按优先级）

1. warp 特化与 persistent 的协议搬进 Mim。
   设想是一个 `ws_fold` 骨架，生产者 warp 与消费者 warpgroup 各走环的一侧，同步用 `mbar_ring` 的 `await`/`release`。
   没有 sm_90 的硬件，只能用 `devsnap.sh` 做设备 IR 与 PTX 的结构对比；交接时刚要开始，还没有改动。
2. 把三套类型打通：解释器的索引、外壳的 `Tile`、`gpu.SharedBuf`。
   最小的一步是让解释器里 `stage` 的结果不再是裸 `Tensor`。
3. 数据 tile 类型的缺口：`layout_expr`（分块、swizzle、填充）、张量核片段及其角色、环的槽位。
   仿射下标没有范围类型，所以绕过 `stored_at` 用逻辑坐标读一个形状碰巧相容的面板不会被拒绝。
4. 带流水环配置（`mimir-tiling128p3`）的 22 例全套：远端机器上一直没有可用的空闲 GPU，bundle 已经在那里。
5. M=49 这类小 M 用例的调度：split-K 对它们是负优化。

已定不做：把缓冲区层解释器并入编译器、提交基准仓库的改动、移动本地的 `feat/gpu-tiling` 分支。

## 5. 不在这个仓库里的东西

下面这些不随 `git push` 走，需要另外拷贝：

- 基准仓库 `gemm-bench-gpu` 里未提交的 GPU 支持（`common/gpu*.py`、`mimir/gpu_compile.py`、`mimir/prepare_gpu.py`、`config/*-gpu.toml`、`tools/` 等，共 33 个改动或新增的文件）和两个结果目录。
  按决定没有提交；没有它们就不能重跑 GEMM 对比。
- 上面列的未跟踪 WIP 文件。
- Claude Code 的记忆（`~/.claude/projects/<主仓库路径的 slug>/memory/`）与会话记录（`~/.claude/projects/<worktree 路径的 slug>/<会话 id>.jsonl`）。
  slug 由绝对路径把 `/` 换成 `-` 得到，所以新机器上路径不同就要放进对应名字的目录，再用 `claude --resume` 接上。

## 6. 写 Mim 时反复碰到的 core 限制

细节在 `docs/tile-law-design.zh-CN.md` §8.3，这里只列会让人卡住的几条：

- 带参数 `nom` 的索引如果不出现在载荷里，就不参与类型相等；要把索引放进单例见证 `«(…)»`。
- 带绑定子的 pack 每次取元素、每次 `lam` 展开都重新实例化；产生类型的函数要加 `@(core.pe.is_closed …)` 过滤器，线性的值只引用一次。
- 异构元组只能用 pack 自己的绑定子做下标，隐式参数要一组给全，否则得到并类型。
- 柯里化 `lam` 的过滤器只管最后一组参数；`(a b: T)` 在柯里化参数里是一个 `«2; T»` 参数。
- 普通 `.mim` 文件里中缀运算符没有绑定，用 `core.nat.*`。
- 嵌套环的状态需要在 `affine.lower_for` 之后做两次 `compile.scalarize`，否则聚合 phi 让 NVPTX 推不出 shared 地址空间。
