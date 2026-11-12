# GPU attention 实验记录

计划见 `docs/flash-attention-plan.zh-CN.md`。机器：A100-PCIE-40GB，PyTorch 2.12.0+cu130。
TFLOP/s 按 `4·BH·L²·D`（causal 减半）计。

## P0 基线（2026-11-12）

PyTorch：`torch_attn.py`（`F.scaled_dot_product_attention`，50 次中位数），完整输出在 `torch_attn.out`。
我们：`kattn.sh <file> H L D`，编译 `-p opt -p gpu -X tensor:tile-fuse=1 … -p nvptx`，nsys 每个 kernel 平均时间之和，驱动 `attn_driver.cpp` 抽样对照 CPU 参考。

BH=32，D=64，非 causal，单位 ms：

| 实现 | kernel 数 | L=1024 | L=2048 |
|---|---:|---:|---:|
| `attn_sdpa.mim`（`tensor.sdpa`，E0：一线程一输出） | 6 | 18.9 | 47.8 |
| `attn_tiled.mim`（QKᵀ、resident online softmax、PV 三个 tile kernel，FP32 SIMT） | 3 | 1.31（6.6 TF） | 4.54（7.6 TF） |
| PyTorch fp32 efficient | 1 | 0.750（11.5 TF） | 1.96（17.5 TF） |
| PyTorch tf32 math（bmm 走 TF32） | 3+ | 0.917 | 3.23 |
| PyTorch fp16 flash | 1 | 0.139（61.7 TF） | 0.241（143 TF） |

- PyTorch 的 efficient 后端打开 TF32 也不变，它是 FP32 SIMT 实现；TF32 tensor core 的 FP32 级参照只能用 math 后端。
- fp16/bf16 flash 在大形状上达到 135–170 TFLOP/s（`torch_attn.out`），是 P2 的目标线。
- `attn_tiled` 三个 kernel 大致各占三分之一：两次 GEMM 和 softmax；S 和 P 都经过全局内存（BH·L² 个 F32，L=2048 时 512 MB 读写）。
- 测量过程中修了两个 TileFuse 的 bug（`lit/gpu/tile_attention.mim`）：
  batched GEMM 的面板放宽把 batch 维设成了"从 0 取满"，head ≥ 1 读到 head 0 的数据；
  softmax 的行统计成员比 work-group 少一维，选面板布局时按 work-group 的维数投影而崩溃。

## 手写 FA2 结构的单 kernel（P1 的上界，2026-11-12）

`flash_tc.py BH L [BC]` 生成，`kflash.sh BH L`（`BC=` 选每块的 key 数）编译、抽样对照并用 nsys 计时，驱动 `flash_driver.cpp`。
D=64，TF32 `mma.m16n8k8`，一个 work-group 64 行 Q、4 个 warp 各 16 行；K/V 块经 `cp.async` 双缓冲进 shared；
S、P、O 和行统计都在寄存器里；行 max 用 quad 内两次 `gpu.shfl_xor`，行和只在最后归约一次。
P 不经 shared 直接作 PV 的 A 操作数：每 8 个 key 内把 2t、2t+1 置换到 t、t+4，V 按同样顺序读。

| BH × L | BC=64 | BC=32 | TFLOP/s（BC=32） | PyTorch fp32 efficient | PyTorch fp16 flash |
|---|---:|---:|---:|---:|---:|
| 32 × 1024 | 0.287 | **0.216** | 39.8 | 0.750 | 0.139 |
| 32 × 2048 | 0.964 | **0.735** | 46.8 | 1.96 | 0.241 |
| 32 × 4096 | 4.09 | **2.52** | 54.6 | 7.46 | 0.851 |
| 128 × 1024 | 1.04 | **0.717** | 48.0 | 1.95 | 0.247 |
| 128 × 2048 | 3.81 | **2.92** | 47.1 | 7.54 | 0.904 |
| 128 × 4096 | — | **10.6** | 52.0 | 29.6 | 3.83 |

- 结果全部正确（每个形状 3 个 head × 4 行 × 64 列与双精度参考比较，容限 2e-3）。
- 比 PyTorch 的 FP32 路径快 2.4–3.5 倍，比 fp16 flash 慢约 3 倍（精度和带宽不同，P2 的目标）。
- 0 字节寄存器溢出，126 个寄存器；BC=64 时 shared 68 KB，每个 SM 只能驻留 2 个 work-group，BC=32 时 34 KB、4 个，快约 30%。
- 新原语 `gpu.shfl_xor`（`llvm.nvvm.shfl.sync.bfly.f32`），测试 `lit/nvptx/shfl_xor.mim`。

## P1 探索：让编译器生成单 kernel（2026-11-12）

- `attn_online.mim`：照搬 `sdpa_online` 的结构（Q 行块的 `tiled`，块内按 key 的 product `tile_fold`，状态 `(max, sum, out)`），把外层 `tiled` 标成 `bind 1`。
  编译成 1 个 kernel，CPU 路径结果正确，GPU 上错误且极慢（2×128 要 22.9 ms）：work-group region 里的 `map_reduce`（两个积、行归约、逐元素）没有再分给 work-item，每个线程重算整块并写同一 buffer。
  region 内的分配只有 TileFuse 对 `tile_req` 的根做，它不认状态是向量的 product fold。
- `attn_elem.mim`：把输出写成对 key 归约的单个 `map_reduce_post`，每个输出元素一份 `(m, l, acc)`，`S = QKᵀ` 作为成员在归约级按块计算，能套用现有 TileFuse；
  代价是 `(m, l)` 与 exp 在 D=64 列上各算一遍。目前在 LowerToMem 里失败：融合的 S tile 存储形状是上界 `(4, 64)`，下游期望运行时长度。

## H200 上重测（2026-11-12，新容器）

机器：H200 NVL（sm_90），驱动 580，PyTorch 2.12.0+cu130；MimIR 用 `build-h200`（clang 20 + gcc15 libstdc++），nvptx 走 LLVM 23。
PyTorch 完整输出在 `torch_attn_h200.out`。我们的 kernel 仍是 sm_80 的 `mma.sync` TF32，没有用 Hopper 专属指令。

D=64，非 causal，单位 ms（括号里 TFLOP/s）：

| BH × L | 手写 FA2 TF32（BC=32） | `attn_tiled` | `tensor.sdpa` | PyTorch fp32 efficient | PyTorch tf32 math | PyTorch fp16 flash |
|---|---:|---:|---:|---:|---:|---:|
| 32 × 1024 | 0.090（95） | 0.517 | 7.39 | 0.328 | 0.381 | 0.0425（202） |
| 32 × 2048 | 0.317（108） | 2.00 | 27.1 | 1.150 | 1.329 | 0.142（242） |
| 32 × 4096 | 1.20（115） | — | — | 9.55 | 10.8 | 0.521（264） |
| 128 × 2048 | 1.21（114） | — | — | 4.39 | 5.09 | 0.532（258） |
| 128 × 4096 | 4.61（119） | — | — | 17.1 | 22.4 | 2.02（272） |

- 手写 FA2 比 PyTorch 的 FP32 路径快 3.6–8 倍；比 fp16 flash（FA2 后端，未用 FA3）慢约 2.2 倍。BC=64 在 H200 上同样慢 20–30%。
- H200 上 PyTorch 的 fp32 efficient 在 L=4096 时反而变慢（14 TFLOP/s），L≤2048 时约 26–31。
- Hopper 的 TF32 峰值要靠 WGMMA 才能接近，`mma.sync` 远达不到；P6（WGMMA、TMA、warp 特化）现在可以实测。

## P1：用 `group_tile_req` 组合出的单 kernel（H200，2026-11-12）

`attn_fa.mim`（`kattn.sh attn_fa.mim BH L 64`，`BM`/`BK` 选块大小）：Q 行块的 `tiled` 绑定到 work-group，块内是按 key 块的 product fold，状态 `(m, l, O)` 在 shared memory；
每步由 `group_tile_req` 组成：`S = Q·K_blkᵀ` 与 `P·V_blk` 走 warp + TF32 mma，行 max/sum、exp、O 的修正分给 work-item。
输入以 `(H·L) × 64` 的二维矩阵给出，块都是二维 `slice`，面板才能按整个 work-group 放宽。

| BH × L | 最初（BK=32） | 现在 BK=32 | 现在 BK=64 | 手写 FA2 | torch fp32 | `attn_tiled`（3 kernel） |
|---|---:|---:|---:|---:|---:|---:|
| 32 × 1024 | 10.8 ms | 0.675 | 0.319 | 0.090 | 0.328 | 0.517 |
| 32 × 2048 | 43.1 ms | 2.65 | 1.02 | 0.317 | 1.15 | 2.00 |
| 32 × 4096 | — | — | 4.07 | 1.20 | 9.55 | — |
| 128 × 1024 | — | — | 1.03 | — | — | — |
| 128 × 2048 | — | — | 4.81 | 1.21 | 4.39 | — |

性能修正（ms 为 32 × 1024、BK=64 时的变化）：
1. fold 输入里的 `slice` 被 `tensor.lower` 降成恒等 `map_reduce_post`，`is_view` 也不认 Var 的投影：每步每个线程串行拷整块 Q。改为视图后 10.8 → 1.73（BK=32）。
2. 逐元素请求的 item 块由 `(1, n)` 改成 `(32, 1)`：同一 warp 原来访问同一列，32 路 bank 冲突。1.31 → 0.599。
3. work-group 内 shared 里的结果块行尾填充 16 字节，逐行归约的冲突降到 4 路。0.599 → 0.509。
   顺带修了向量填充的对齐判断：以前用逻辑行长而非存储行长，填充不是 16 字节倍数时结果错误。
4. 最后的请求直接产出 `ext` 形状，不再 `reshape`：`tensor.lower` 把形状不能证明相同的 reshape 降成拷贝，结果先进 shared 再拷出。
   shared 从 115.7 KB 降到 98 KB，每个 SM 从 1 个块变成 2 个。0.509 → 0.401。
5. product fold 中只被自己的更新逐元素读的状态（l、O）原地更新，不再双缓冲（`pointwise_step`）；m 仍被 c 读，保留双缓冲。
   shared 降到 81.7 KB。0.401 → 0.373。
6. 逐元素读一个 work-group 请求结果、且该结果的其余读者都先于它运行的请求，直接写进那块缓冲（`reusable_input`）：P 复用 S。
   shared 降到 64 KB，每 SM 3 个块。32 × 2048：1.61 → 1.36；32 × 1024 因 512 个块按 3 个/SM 只有 1.3 波，0.373 → 0.407。
7. PV 直接累加进 O：`o = o·c` 原地写进状态，再 `o = map_reduce_acc (mac, o) (P, V)` 让 tensor core 从 O 的元素起累加。
   通用部分：`map_reduce_acc`（张量初值的 `map_reduce`）；`producer_init` 按语义比较初值映射（`Linear::expanded`）；
   tensor-core fold 的累加器分片从初值张量装载（`load_fragments`）；`reusable_input` 也复用 fold 状态分量；
   不读累加器的步骤（只经已完成的请求间接用到）可以原地写。测试：`lit/{gpu,nvptx}/group_tile_req_acc.mim`。
   少一块 PV 结果和一遍逐元素读写。32 × 1024：0.407 → 0.319；32 × 2048：1.36 → 1.02。
- 占用率的影响：把同一 kernel 启动时的 shared 人为调到 120 KB（每 SM 1 个块），32 × 1024 从 0.371 变成 0.675 ms。
- BK=64 时逐元素请求必须是 `(32, 1)`：`(1, 16)` 会切出 256 个 item，多于 128 个线程，barrier 落进发散分支。
- 消融（BK=64，修正 1 之后）：PV 约 0.3 ms，行统计约 0.44 ms，O 的修正约 0.5 ms；mma 的 k 块大小和 cp.async 流水线影响都在 ±10% 内。

- 结果正确（`lit/gpu/attention_fold.mim` 与 nvptx 版本），一个 kernel，S 与 P 不落全局内存；现在比手写版本慢 3.2–4 倍；L≤2048 时与 torch fp32 相当或略快，L=4096 时快 2.3 倍。
- 为此修的问题：work-group 层级的分配（请求结果、fold 状态）进 shared；原地 work-group 级里 item/warp region 之后的 barrier；
  `root_contraction` 接受带偏移的映射和转置读的 B；`group_slice` 在切片自己的坐标系里放宽（batch 维、k 方向的起点）；
  `reshape(slice)` 作 region 输入时是视图，`get_fused` 读穿只改单位维的 reshape。
- 剩余的性能问题：每个 fold 步约 40 道 barrier（mma 的每个 k 块两道）；行归约每线程一整行、只用一半线程（应改为 warp shuffle）；Q 面板每步重填；S 与 O 仍经 shared 往返（手写版本全在寄存器里）。
