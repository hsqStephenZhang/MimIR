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

## P1：用 `group_tile_req` 组合出的单 kernel（2026-11-12）

`attn_fa.mim`（`kattn.sh attn_fa.mim BH L 64`，`BM`/`BK` 选块大小，现在用 BK=64）：Q 行块的 `tiled` 绑定到 work-group，块内是按 key 块的 product fold，
状态 `(m, l, O)` 在 shared memory；每步由 `group_tile_req` 组成：`S = Q·K_blkᵀ` 与 `O += P·V_blk` 走 warp + TF32 mma，
行 max/sum 每行两个 lane，exp 与 O 的修正分给 work-item。
输入以 `(H·L) × 64` 的二维矩阵给出，块都是二维 `slice`，面板才能按整个 work-group 放宽。

**机器变了**：这一轮中途容器从 H200 NVL（sm_90）迁到了 RTX PRO 6000 Blackwell Server Edition（sm_120，每 SM 100 KB shared，单块最多 99 KB），
下面各步骤的前后对比都是同一台机器上连续测的，但并非所有步骤都在同一台机器上；最终对照全部在 RTX PRO 6000 上重测
（PyTorch 输出在 `torch_attn_rtx6000.out`）。我们的 kernel 仍按 sm_90 编译，靠 PTX JIT 运行。

RTX PRO 6000，D=64，非 causal，单位 ms：

| BH × L | `attn_fa`（编译器生成） | 手写 FA2（BC=32） | PyTorch fp32 efficient | PyTorch fp16 flash |
|---|---:|---:|---:|---:|
| 32 × 1024 | 0.244 | 0.100 | 0.228 | 0.050 |
| 32 × 2048 | 0.827 | 0.305 | 0.781 | 0.128 |
| 32 × 4096 | 3.13 | 1.10 | 2.64 | 0.455 |
| 128 × 1024 | 0.787 | 0.300 | 0.725 | 0.131 |
| 128 × 2048 | 3.30 | 1.09 | 2.62 | 0.428 |

比手写版本慢 2.4–3 倍，比 PyTorch fp32 efficient 慢 7–26%。最初（H200）是 10.8 ms（32 × 1024）。

性能修正，按顺序（括号里是同机前后对比）：
1. fold 输入里的 `slice` 被 `tensor.lower` 降成恒等 `map_reduce_post`，`is_view` 也不认 Var 的投影：每步每个线程串行拷整块 Q。改为视图（32 × 1024，BK=32：10.8 → 1.73）。
2. 逐元素请求的 item 块由 `(1, n)` 改成 `(32, 1)`：同一 warp 原来访问同一列，32 路 bank 冲突（BK=64：1.31 → 0.599）。
3. work-group 内 shared 里的结果块行尾填充 16 字节，逐行归约的冲突降到 4 路（0.599 → 0.509）。
   顺带修了向量填充的对齐判断：以前用逻辑行长而非存储行长，填充不是 16 字节倍数时结果错误。
4. 最后的请求直接产出 `ext` 形状，不再 `reshape`：`tensor.lower` 把形状不能证明相同的 reshape 降成拷贝，结果先进 shared 再拷出（0.509 → 0.401）。
5. product fold 中只被自己的更新逐元素读的状态（l、O）原地更新，不再双缓冲（`pointwise_step`）；m 仍被 c 读，保留双缓冲（0.401 → 0.373）。
6. 逐元素读一个 work-group 请求结果、且该结果的其余读者都先于它运行的请求，直接写进那块缓冲（`reusable_input`）：P 复用 S（32 × 2048：1.61 → 1.36）。
7. PV 直接累加进 O：`o = o·c` 原地写进状态，再 `o = map_reduce_acc (mac, o) (P, V)` 让 tensor core 从 O 的元素起累加。
   通用部分：`map_reduce_acc`（张量初值的 `map_reduce`）；`producer_init` 按语义比较初值映射（`Linear::expanded`）；
   tensor-core fold 的累加器分片从初值张量装载（`load_fragments`）；`reusable_input` 也复用 fold 状态分量；
   不读累加器的步骤（只经已完成的请求间接用到）可以原地写。测试：`lit/{gpu,nvptx}/group_tile_req_acc.mim`（32 × 2048：1.36 → 1.02）。
8. 行 max/sum 每行由相邻 2 个 lane 分担，各折叠半行，再用 `shfl_xor` 合并（通用层级 `item_lanes`，`bind 4`；测试 `lit/{gpu,nvptx}/group_tile_req_lanes.mim`）（1.02 → 0.932）。
   每行 4 个 lane 需要 256 个线程，超出块大小，目前不支持。
9. 不同 work-group 构造的面板从不同时存活（构造不嵌套、各自在 barrier 后进入；LowerToMem 检查不嵌套），`gpu.panel_scope` 标记后由 nvptx 重叠放置。
   只在不重叠的布局超过 48 KB 静态上限时重叠：重叠后 LLVM 看不出面板互不别名，小 kernel 反而慢约 3%。
   mma 的 k 块从 8 改成 16：shared 从 55.3 KB（每 SM 1 块）降到 45.6 KB（2 块）（0.934 → 0.827）；k=32 与 cp.async 流水线仍只有 1 块，更慢。
- 占用率是主要因素：每 SM 只有 100 KB，44.5 KB 以下才能放 2 个块，32 KB 以下才能 3 个。
- 结果正确（`lit/gpu/attention_fold.mim` 与 nvptx 版本），一个 kernel，S 与 P 不落全局内存。
- 为此修的问题：work-group 层级的分配（请求结果、fold 状态）进 shared；原地 work-group 级里 item/warp region 之后的 barrier；
  `root_contraction` 接受带偏移的映射和转置读的 B；`group_slice` 在切片自己的坐标系里放宽（batch 维、k 方向的起点）；
  `reshape(slice)` 作 region 输入时是视图，`get_fused` 读穿只改单位维的 reshape。
- 剩余的性能问题：S（17 KB）与 O（16 KB）在 shared 里，手写版本全在寄存器里（shared 35 KB 只放 K/V）；Q 面板每步重填；每个 fold 步十余道 barrier。

## warp 拥有的区域（`attn_warp.mim`，RTX PRO 6000，2026-11-12）

attention 写成 work-group → 每个 warp 拥有 16 行（`tensor.bind 5`）→ 按 key 的 fold；区域里的张量是 warp 值，
LowerToMem 把它们翻译成 `tensor.warp` 的 Mim 函数（见 `docs/warp-schedule.zh-CN.md`），S、P、O 与行统计都在寄存器里。
K/V 目前由每个 lane 直接从全局内存读，还没有经 shared 协作装载与预取。

| BH × L | `attn_warp` | `attn_fa` | 手写 FA2 | PyTorch fp32 efficient |
|---|---:|---:|---:|---:|
| 32 × 1024 | 0.158 | 0.244 | 0.100 | 0.228 |
| 32 × 2048 | 0.590 | 0.827 | 0.305 | 0.781 |
| 32 × 4096 | 2.56 | 3.13 | 1.10 | 2.64 |
| 128 × 1024 | 0.640 | 0.787 | 0.300 | 0.725 |
| 128 × 2048 | 3.66 | 3.30 | 1.09 | 2.62 |

### 面板由 work-group 协作装载（H200 NVL，2026-11-13）

容器回到了 H200（tyr2）。`attn_warp.mim` 的 step 用 `tensor.stage` 把 K/V 块装进 shared：整个 work-group 协作填充，
warp 从 shared 读 B 分片；面板行尾填充 16 字节（否则按转置读 K 有 8 路 bank 冲突）。

| 32 × 2048 | ms |
|---|---:|
| `attn_warp`，K/V 每 lane 直接读全局 | 1.032 |
| `attn_warp`，K/V 经 shared 面板（无填充） | 2.094 |
| `attn_warp`，K/V 经 shared 面板（填充） | 0.833 |
| `attn_fa` | 1.096 |
| 手写 FA2（`flash_tc`）/ warp 代数版（`flash_warp`） | 0.317 / 0.316 |

剖析（`-X nvptx:profile=1`，线程 0）：每 64 个 key 约 8.3k 周期，手写约 5k；两次面板填充约占 20%（还没有预取）。
寄存器 184 个，每 SM 2 个块，手写 128 个、4 个块。BK = 32 时寄存器降到 140，但步数加倍，1.122 ms。

### Q 的分片在循环外装载一次（H200）

warp 拥有的 fold 在循环前扫描 step：A 操作数来自内存且不随步变的乘积（S = Q·Kᵀ 的 Q），每个 lane 的 A 分片在循环外装载，
由循环原样携带（这样装载才落在循环之前），step 里改读这些值。

| BH × L | `attn_warp`（BK=32） | 手写 FA2 | 比值 |
|---|---:|---:|---:|
| 32 × 1024 | 0.159 | 0.089 | 1.8 |
| 32 × 2048 | 0.534 | 0.317 | 1.7 |
| 32 × 4096 | 2.05 | 1.20 | 1.7 |
| 128 × 1024 | 0.552 | 0.323 | 1.7 |
| 128 × 2048 | 2.01 | 1.21 | 1.7 |

32 × 2048：BK = 32 从 1.126 降到 0.534，BK = 64 从 0.835 降到 0.651。
