# FlashAttention 在 MimIR GPU 上的计划（2026-11-12）

目标机器是 A100（sm_80）。
FlashAttention-3 的核心手段（WGMMA、TMA、`setmaxnreg` 的生产者/消费者 warp 特化、FP8）都要 Hopper（sm_90），
所以计划分两层：在 A100 上做 FA2 级别的单 kernel 融合，并移植 FA3 中不依赖 Hopper 的调度思想；Hopper 专属部分只做设计。

## 0. 现状

- 加载 gpu 插件后，attention 的 group/online 规则整体关闭（`group_rules.cpp`、`group.cpp`、`normalizers.cpp` 中的 `is_loaded("gpu")`），
  GPU 上的 `tensor.sdpa` 是 6 个互不融合的 kernel；`bench/gpu_fusion/attn_tiled.mim` 是 3 个 kernel，S 矩阵落到全局内存。
- 可复用：带 `merge_with` 的 online (m, l) 折叠、`sdpa_online` 规则、TF32 `mma.m16n8k8`、`ldmatrix`、`cp.async` 多级环、相位类型与 `check_sync`、
  逐元素生产者融进面板填充（`tensor.get_fused`）。
- 缺：F16/BF16 tensor core、warp 内 shuffle 归约、fold 的 step 里串两个矩阵乘、fragment 的布局类型（G10）。

## 1. FA3 手段在 A100 上的对应

| FA3 | sm_90 机制 | A100 |
|---|---|---|
| 异步 warpgroup MMA | WGMMA | `mma.sync` m16n8k16，靠指令交错 |
| TMA 与生产者 warp | TMA、`setmaxnreg` | `cp.async` 多级环 |
| warpgroup ping-pong | named barrier 交替 GEMM 与 softmax | `bar.sync id, n` 分两组 warp |
| warpgroup 内重叠 | 先发第 j+1 块的 WGMMA 再做第 j 块的 softmax | fold 循环体两级旋转的软件流水 |
| FP8 | FP8 WGMMA | 不可用 |

## 2. 阶段

- **P0 基线**：形状 B·H ∈ {32, 128}，L ∈ {1k, 2k, 4k, 8k}，D ∈ {64, 128}，causal 与否。
  对照 PyTorch `F.scaled_dot_product_attention`（fp16/bf16 flash、fp32、tf32）与我们现有的 6-kernel 和 3-kernel 版本。
  指标：TFLOPS、kernel 数、S 是否落到全局内存。交付 `bench/gpu_attn/`。
- **P1 单 kernel 融合（TF32/FP32）**：`sdpa_online` 在 GPU 上降成一个 `tile_req`，根是按 S 维分块的 fold，状态 `(m, l, O_tile)`，
  step 内 `S_blk = Q_tile·K_blkᵀ` → online softmax → `O += P·V_blk`；Q 面板常驻，K/V 每块 stage，S/P 不落全局内存；
  `(m, l, O)` 的合并律由 `merge_with` 声明。
- **P2 F16/BF16 tensor core**：`gpu.mma_f16/bf16`（m16n8k16），`ldmatrix` b16（含 trans）；
  P 在寄存器里复用：`Frag (shape, layout, T)` 与 `c_to_a` 公理把"累加器 C 布局打包后即第二个 GEMM 的 A 布局"写进类型。
- **P3 warp 级归约与相位**：quad 内 `gpu.shfl_xor` 蝶形归约行 max/sum，引入 `WM` token，发散检查转到类型上。
- **P4 FA3 调度思想**：fold 循环体两级旋转（先发 `S_{j+1}` 的 mma 再做 `P_j`），作为层级参数由 Mim 函数给出轮转计划；
  warp 分组 ping-pong 用 named barrier，每组一个构造 `g`，`check_sync` 扩展为两组交替推进；A100 上先测再定是否保留。
- **P5 causal 与边界**：按块跳过、对角块逐元素 mask（块数依赖 Q 块号的动态上界）；不整除长度用 padding 面板加 mask；GQA/MQA 写进访问映射。
- **P6 Hopper（只做设计）**：WGMMA、TMA、mbarrier 的 phase 位与超步代数 `e` 对应、生产者/消费者分属两个构造（需要循环携带 token）、FP8。
- **P7 反向**：重算 S、dQ 的 atomic 累加、dK/dV 按块归约。

## 3. 每阶段

- 正确性：数值对照参考实现（fp16 与 tf32 两档容限），lit 的 IR 检查与 nvptx 执行测试。
- 形式化：新原语（`Frag` 布局转换、shuffle、named barrier、旋转计划）写进形式化文档，列出前提与可信边界。
- 性能：在 P0 的形状集上重测，更新 `bench/gpu_attn/RESULTS.zh-CN.md`。

## 4. 风险

- 精度主线：P1 用 TF32 验证结构，P2 再上 F16/BF16。
- ping-pong 在 A100 上收益不确定，先做软件流水。
- P1 的 TileFuse 改动最大；若与单收缩假设冲突太多，先在 Mim 里手写 FA2 kernel 作为性能上界（同 `gemm_tc.py` 的做法）。
- 没有 H100，P6 只能写设计和类型原型。
