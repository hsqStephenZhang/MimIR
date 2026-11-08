# GPU 上的 fusion 与 softmax 结构（2026-11-08）

机器：GTX 1650 SUPER（sm_75）。程序：`softmax.mim`（12×1024×1024 行 softmax）、`attn.mim`（H=12，L=512，D=64 的 `tensor.sdpa`）。
编译 `mim -p opt -p gpu <prog>.mim -p nvptx`，默认流水线（E0 路径：每个 `map_reduce_post` 一个 kernel，一线程一个输出点，归约在线程内串行）。
时间为 nsys 的 kernel 平均值；端到端含每个 op 的 H2D/D2H 拷贝与分配。

| 程序 / 变体 | kernel 数 | kernel 时间合计 | 端到端 | PyTorch eager |
|---|---:|---:|---:|---:|
| softmax 默认 | 4（max 2.61，exp−max 0.63，sum 2.61，div 0.63 ms） | 6.5 ms | 229 ms | 0.619 ms |
| softmax `-X tensor:group=0` | 4，同上 | 6.5 ms | 229 ms | — |
| attention 默认 | 6（QKᵀ 5.98，PV 0.82，max/sum 0.39/0.38，两次逐元素 0.16） | 7.9 ms | 171 ms | 0.704 ms（两次 matmul + softmax）/ 0.875 ms（F.sdpa） |
| attention `group=0` | 6，同上 | 7.9 ms | 177 ms | — |
| attention `-X tensor:attention-online=1` | 6，同上 | 7.9 ms | 175 ms | — |

- 三个 attention 变体的 kernel 完全相同：`tensor.group`（含 online 规则所在的 `group_attention`）在加载 gpu 插件时整体关闭（`normalizers.cpp`、`group_rules.cpp` 里 `is_loaded("gpu")` 直接返回），所以 GPU 上既没有 group 级 fusion 也没有 online 结构。
- softmax 的 4 个 kernel 对应 max → exp(x−max) → sum → div；`exp` 的结果被 sum 和 div 两处使用，普通 map-into-reduce fusion 无法不重算地合并。
- 归约 kernel 慢的原因不是算力：grid 12 × block 1024，一线程一行串行 1024 步，一个 warp 的 32 个线程读 32 行相隔 4 KB 的地址，每个请求 32 个 sector，DRAM 读 343 MB（输入只有 48 MB，7 倍放大）。
- 逐元素 kernel 0.63 ms 已接近带宽（读 48 MB 写 48 MB）；PyTorch 的单 kernel softmax 0.62 ms 只读写各一遍。
- 要在 GPU 上做 fused/online softmax，缺的是跨线程归约（设计文档 §2.6 的 R8）：一个 block 负责一行（或几行），线程各持一段，max/sum 用 shared/warp 归约合并，exp 与 div 在寄存器内完成；online 形式则是单遍的 (m, l) carry 合并，同样需要跨线程 combine。

## R8：归约级绑定到 work-item（同日）

| 程序 | 配置 | kernel | PyTorch |
|---|---|---:|---:|
| `rowsum.mim`（12×1024×1024 → 12×1024） | group `(1, 8)`，归约级 `(0, 0, 4)` bind=2：256 item 各折 4 个元素，shared 里 256×8 个 partial，8 个 item 各合并一个输出 | **0.291 ms** | 0.286 ms（`x.sum(-1)`） |
| E0 路径的同一归约 | 一线程一行 | 2.61 ms | |
| `softmax_r8.mim`（三个 `tile_req`：max、Σexp(x−max)、exp(x−max)/s） | 同上 + map 级 `(1, 8, 4)` | 0.286 + 0.295 + 0.682 = **1.26 ms** | 0.619 ms |
| E0 路径 softmax | 四 kernel | 6.5 ms | |

ncu（rowsum）：DRAM 读 51 MB（输入 48 MB）、每请求 16 sector、2 次 barrier、barrier stall 45%（第二个 item 循环只有 8 个 item 有活）。
map kernel 0.682 ms 读 48 MB 写 48 MB，已接近带宽；三趟一共读 144 MB、写 48 MB，这就是和 PyTorch 单趟的差距来源。

## online softmax（同日）

| 程序 | kernel | PyTorch |
|---|---:|---:|
| `softmax_online.mim`：pass 1 `(max, sum)` online fold，R8 + 显式 merge；pass 2 归一化 | 0.459 + 0.692 = **1.15 ms** | 0.619 ms |
| `softmax_r8.mim`（三趟） | 1.26 ms | |
| E0 路径 | 6.5 ms | |

- pass 1：读 50 MB，2750 万条指令（每元素两次 `exp`：修正项和元素项），FMA 管线 41%，XU 21%，barrier stall 46%（merge 阶段只有 8 个 item 在干活，各串行扫 256 个 partial）。
- pass 2：读 51 MB 写 52 MB，674 µs ≈ 153 GB/s，带宽上限。
- 改进方向：按 block 做 online 更新（每 block 一次修正而不是每元素）、merge 用树形/warp shuffle、以及最终的整行常驻单 kernel。

## 整行融合单 kernel（同日）

| 程序 | 结构 | kernel | PyTorch |
|---|---|---:|---:|
| `softmax_fused.mim`：`tile_req (y, (mx, s), …)`，group `(1, 8, 1024)`，item `(1, 8, 4)` | TileFuse 把 mx、s 两个归约 member 拆成 group 体里的 R8 fold（`L1 red … ks=4 bind=2`），s 的 step 读 mx 的共享 tile，item 级读两者；1 kernel、0 malloc、5 barrier | **1.20 ms** | 0.619 ms |
| `softmax_fused_online.mim`：member 是 `merge_with merge` 标注的 (m, l) online fold | 1 kernel、0 malloc、3 barrier | **0.92 ms** | |
| `softmax_online.mim`（两趟、两 kernel） | | 1.15 ms | |
| `softmax_r8.mim`（三趟） | | 1.26 ms | |

ncu（`softmax_fused`）：DRAM 200 MB（x 读三遍 50 MB × 3 + 写 50 MB），DRAM 吞吐 90% 峰值，long scoreboard stall 64%，barrier stall 9%，shared bank conflict 25.8 万，49 寄存器，16.5 KB 静态 shared。
即两个版本都在各自读次数的带宽上限；PyTorch 的 0.62 ms 是读一遍。group 的 8 行是 32 KB，并发 block 的工作集超过 1 MB L2，三趟之间没有缓存命中。下一步是把 group 的输入 tile 常驻 shared/寄存器（G4）。

## 常驻单 kernel（同日，G4）

| 程序 | 结构 | kernel | PyTorch |
|---|---|---:|---:|
| `softmax_resident.mim`：group `(1, 4, 0)` promote 输入 0，item `(1, 4, 4)`，member 是 `merge_with` 的 online fold | 4 行 × 1024 的 x 协作填进 shared（16 KB），(m, l) fold 和归一化都读 shared；R8 partial 树形合并；1 kernel、0 malloc、12 barrier | **0.611 ms** | 0.619 ms |
| `softmax_resident8.mim`：group 8 行，item `(1, 8, 8)`（128 item） | 40 KB shared，1 block/SM | 1.547 ms | |
| `softmax_fused_online.mim`（非常驻，树形 combine） | | 0.911 ms | |
| `rowsum.mim`（树形 combine） | | 0.286 ms | 0.286 ms |

ncu（`softmax_resident`）：DRAM 100 MB、89% 峰值带宽，barrier stall 8%（串行 combine 时 61%），long scoreboard 31%，38 寄存器，24.6 KB 静态 shared。到此 softmax 在这块卡上就是带宽上限。
