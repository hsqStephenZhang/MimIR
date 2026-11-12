# Rule 识别后的 CPU attention 联合调度

2026-11-12，本机 i5-14600KF，FP32 / AVX2。实现提交 `95153efe99`，静态形状 guard `06305c9f61`。

## 识别与调度链路

复用 `tensor.raise_attention` 对分解图的识别，得到 `tensor.sdpa` / `tensor.sdpa_masked`，
再由 `tensor.group_attention` 应用新的 declarative CPU schedule rules。
调用方既可以写完整的分解图，也可以直接使用 SDPA；无需手工构造行块区域。
新增回归覆盖分解图识别、masked/unmasked、多线程、尾块、全 mask 行和动态形状不匹配。

规则生成 `sdpa_cpu`，随后降低为：

1. 按 `(head, query-row-block)` 划分共同区域；多线程时以 CPU `bind 4` 绑定，单线程为普通循环。
2. 在区域内计算 QK、scale/mask、行最大值、未归一化 exp 和行和。
3. 计算 `EV`，最后对输出按行除以和；不再构造完整的归一化 probability 张量。
4. 两个 contraction 显式使用 `tile_req + cpu_default_levels`，复用现有
   Par/Red/Pack/Vec、MR/NR/MC/NC/KC 与 CPU loop-order 选择，没有嵌套 OpenMP。

这版是 **CPU 行块联合调度**：每块仍覆盖完整 key 序列，临时 score/exp 为行块大小，
工作集随 `BM*S` 增长。它尚未采用沿 key 分块的 online 累加。
这里借鉴 IREE LLVMCPU 的整体区域分块和归一化后移；没有将 IREE 当作性能对照后端。

## 启用与范围

```sh
build/bin/mim -p opt -p omp input.mim -p ll \
  -X tensor:attention-cpu=1 -X tensor:attention-cpu-rows=32 -X omp:threads=6
```

- `attention-cpu=1` 启用联合调度，默认仍为0；`attention-cpu-rows` 默认32。
- 自动规则当前针对 `math.mode.contract` 的 FP32 SDPA、静态形状、非空 key/output feature。
  动态形状不能满足 guard 时保留原路径。`group=0` 或 `attention=0` 不应用该规则。
- `attention-online=1` 仍选择此前的实验性 online 路径，优先于该开关。
- 当前单线程部分尺寸有回退，因此没有整体替换默认路径；BM=32 是这次测量的配置，并非调优结论。
- 显式提供的 mask 是输入。benchmark 中 causal mask 由 `tensor.generate` 产生，
  该输入仍可能在进入区域前完整物化，尚未把 mask 生成沉入行块。

## 正确性、内存及调度验证

- 88个原生内核（原路径/联合路径×11形状×4线程配置）全部与独立 FP64 参考比较通过；
  包括两个 Torch 后端在内，最大绝对误差约 `1.67e-7`。
- 联合路径44个配置全部通过独立 ASan/LSan，未使用 suppression；每个预热10次、重复20次。
- 6个原生探针在预热10次后连续调用1000次，净分配增长均为0 bytes。
- 多线程代表用例的线程team和实际绑核均已核对：原路径每次2个 launch，联合路径1个，
  4/6/8个线程均实际参与；单线程两条路径均无 OpenMP launch。
- 全套260 CTest和825 lit通过（8 unsupported）；补充静态 guard 后260 CTest与14项 attention相关测试再次通过。
- guard仅改变规则适用范围。重新编译静态代表用例后，入口及内核的归一化反汇编指令一致；
  ELF `.text` 因符号相关地址而不同，记录见 `static-guard-equivalence.json`。

## 持续计时结果

所有后端为常驻进程，至少0.2秒预热，9轮随机交错顺序，每个连续样本至少100 ms，报告中位数。
编译、数值/线程/内存校验全部结束后，串行测1/4/6/8线程。
MimIR的中间分配和返回张量释放都计时；Torch使用原生ATen循环计时。
测量给定Q/K/V的SDPA核心，不含QKV/输出投影。
Torch SDPA规则尺寸走CPU flash attention；D32/N48非整齐用例回退math。
Torch显式组合为 `bmm → scale/mask → softmax → bmm`，固定mask在计时外建立。

H=8、M=S=128、D=N=64，non-causal，单位 **ms**：

| 线程 | 原路径 | 联合调度 | Torch SDPA | Torch 显式组合 | 相比原路径 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.5256 | 0.4466 | 0.3110 | 0.4325 | 1.18× |
| 4 | 0.3573 | 0.1125 | 0.0930 | 0.2024 | 3.18× |
| 6 | 0.3338 | 0.0864 | 0.0702 | 0.1936 | 3.87× |
| 8（6P+2E） | 0.3911 | 0.1648 | 0.1236 | 0.2336 | 2.37× |

H=8、M=S=512、D=N=64，non-causal，单位 **ms**：

| 线程 | 原路径 | 联合调度 | Torch SDPA | Torch 显式组合 | 相比原路径 |
|---|---:|---:|---:|---:|---:|
| 1 | 8.3286 | 7.0881 | 4.4742 | 7.2093 | 1.18× |
| 4 | 7.1287 | 1.7716 | 1.2574 | 5.1868 | 4.02× |
| 6 | 5.9497 | 1.2471 | 0.8825 | 4.6753 | 4.77× |
| 8（6P+2E） | 7.8526 | 2.5705 | 2.0092 | 6.2402 | 3.05× |

11个形状的几何平均（收益大于1表示联合路径更快）：

| 线程 | 联合/原路径收益 | 联合路径相对自身单线程加速 | 原路径相对自身单线程加速 |
|---|---:|---:|---:|
| 1 | 0.95× | 1.00× | 1.00× |
| 4 | 2.54× | 3.54× | 1.32× |
| 6 | 2.97× | 4.38× | 1.40× |
| 8 | 2.22× | 2.86× | 1.22× |

单线程存在回退，例如：

| 用例 | 原路径（ms） | 联合（ms） | 联合/原路径耗时 |
|---|---:|---:|---:|
| h8_s128_d128 | 0.7358 | 0.7879 | 1.07× |
| h4_cross_128_512 | 1.1758 | 1.8152 | 1.54× |
| h8_decode_512 | 0.0830 | 0.2408 | 2.90× |

8线程包含6P+2E，不能视作8个同质P核；两条MimIR路径和Torch均出现相对6P的退化。
CPU行块并行化解决了原先串行softmax的扩展限制，但没有消除块内物化、packing开销，
也没有完成按尺寸/线程选择BM和适用路径的成本模型。对这些阶段未做定量性能归因。

## 配置与复现

与上次泄露修复后的报告一致：Haswell/AVX2、MR6/NR16/MC320/NC104512/KC320、order1，
`btensor:registers=128`。MimIR和Torch共用wheel内的GNU libgomp；
PyTorch2.12.0+cpu、MKL2024.2。CPU0为主线程，1/4/6线程分别用1/4/6个P核，
8线程加CPU12/13两个E核。PASSIVE等待、spin0、关闭动态线程和嵌套线程。
未锁频，governor为powersave、正常turbo；波动记录见 `timing-variation.json`。
所有88个MimIR二进制均检查无ZMM/EVEX。

```sh
PY=/workspaces/ml-compiler/gemm-bench/.venv/bin/python
ROOT=/tmp/mimir-attention-rule-bench
$PY bench/cpu_tiling/attention_cpu.py tools --root "$ROOT" --variants mimir_default mimir_joint
for t in 1 4 6 8; do
  $PY bench/cpu_tiling/attention_cpu.py prepare --root "$ROOT" --threads "$t" --variants mimir_default mimir_joint
  $PY bench/cpu_tiling/attention_cpu.py validate --root "$ROOT" --threads "$t" --variants mimir_default mimir_joint
done
$PY bench/cpu_tiling/attention_audit.py --root "$ROOT" --variants mimir_default mimir_joint
# 完成所有编译和校验后串行计时：
for t in 1 4 6 8; do
  $PY bench/cpu_tiling/attention_cpu.py run --root "$ROOT" --threads "$t" --variants mimir_default mimir_joint
done
```

`mimir_default`显式传入`attention-cpu=0`，`mimir_joint`传入1；两者均关闭旧online路径。
原始样本、FP64校验、worker审计、LSan日志、命令和二进制/IR哈希均随报告保存。
性能使用`95153efe99`中的实现。编译器构建于提交前，metadata保留当时base HEAD及dirty列表；
`compiled-source-state.json`和`compiler.patch`标识准确源码，`source-state.json`另记录最终guard提交。
最终guard只限制动态形状匹配，没有改变本次静态样本所用的调度和计算。
