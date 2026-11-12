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
