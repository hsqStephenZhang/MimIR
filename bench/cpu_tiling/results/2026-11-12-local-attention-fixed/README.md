# 修复 OpenMP 缓冲区泄露后重测 CPU attention

2026-11-12，Intel Core i5-14600KF，FP32 / AVX2，本机。
修复提交：`6afcdb0144`，分支 `feat/cpu-multilevel-omp`。
测量给定 Q/K/V 的完整 SDPA 核心 `softmax(QKᵀ / sqrt(D) + mask) V`，不含 QKV/输出投影。
H 是 batch×heads，形状为 Q[H,M,D]、K[H,S,D]、V[H,S,N]。

## 泄露根因和修复

`omp.outline` 将并行体提升为外部内核，`omp.launch` 通过符号名引用它。
`mem::EscapeAnalysis` 原先把它视作可能保留参数的未知内存调用。
attention 的两个中间缓冲区因此被标记为逃逸，`mem.free_locals` 未生成释放。
旧版 H=8,M=S=128 每次泄露 1 MiB；H=8,M=S=512 每次泄露 16 MiB。
旧版证据保留在 [上次报告](../2026-11-12-local-attention/README.md)。

本次增加可选的 `mim_mem_borrows` 插件钩子：操作必须同步完成，不保留或释放参数中的句柄，且只返回内存。
OpenMP 钩子解析实际内核并检查其可达代码：允许读取、地址计算和标量写入，
拒绝保存指针、将指针转成整数、未知的指针调用、捕获的未知回调，以及释放捕获指针的内核。
内核自身分配的局部缓冲区可以显式释放。
未能证明的内核仍保守处理；没有将全部 OpenMP 调用声明为不逃逸。
调用继续参与最后使用位置计算，释放安排在同步调用及其他后续使用完成之后。

## 内存和正确性验证

- 44 个默认 attention 配置（11 形状×1/4/6/8线程）全部通过 ASan/LSan；
  另有4个 online 对照，共48个独立进程，每个预热10次、检查20次。**未使用 LSan suppression**。
- 6 个原生长运行探针（各线程的 H8/S128，以及4线程 S512、causal S256），
  预热10次后连续调用1000次，`mallinfo2.uordblks+hblkhd` 的净增长均为 **0 bytes**。
  每次检查数值并只释放返回张量，中间缓冲区由生成代码释放。
- 全部88个原生内核（默认/online×11形状×4线程配置）与独立 FP64 稳定 softmax 参考比较；
  所有 MimIR 和 Torch 输出检查通过，具体误差见各线程 `validation.json`。
- 新增并行 ordinary/causal attention 的重复调用 LSan 回归；新增保存指针、未知调用、
  指针转整数、内核消费指针等反例，检查没有错误插入释放。
- 260 CTest、823 lit 测试通过；8项 unsupported。

`sanitizer.json` 保留48个编译/运行命令与退出码，`sanitizer/*.lsan.log` 为原始输出；
`native-memory.json` 是1000次调用的未插桩检查。ASan 拦截分配器，因此堆增长数字以未插桩探针为准。

## 重测结果

所有后端使用常驻进程，至少0.2秒预热，9轮交错随机顺序，报告每轮每调用时间的中位数。
每个连续计时样本至少100 ms，无泄露限额、无逐样本重启进程。
计时在编译、正确性、线程审计和内存检查全部结束后串行执行。
原生 C++ 循环计时，MimIR 包括中间分配和返回张量的释放；Torch 使用原生 ATen 计时。
输入准备和输出比较不在计时区间。

`torch_sdpa` 对应 PyTorch 默认 scaled_dot_product_attention，规则尺寸走 CPU flash attention，
D=32/N=48 的非整齐用例回退 math，见 profiler 审计。
`torch_explicit` 为 `bmm → scale/mask → softmax → bmm`；固定 causal mask 在计时前建立。
`mimir_default` 是 multi-level tiling 的默认分解图；两次 matmul 并行，softmax 仍是顺序循环。

H=8、M=S=128、D=N=64，non-causal，单位 **ms**：

| 线程 | MimIR 默认 | Torch SDPA | Torch 显式组合 | MimIR 相对单线程 |
|---|---:|---:|---:|---:|
| 1 | 0.5224 | 0.3108 | 0.4338 | 1.00× |
| 4 | 0.3574 | 0.0934 | 0.2019 | 1.46× |
| 6 | 0.3351 | 0.0719 | 0.1899 | 1.56× |
| 8（6P+2E） | 0.3904 | 0.1238 | 0.2190 | 1.34× |

H=8、M=S=512、D=N=64，non-causal，单位 **ms**：

| 线程 | MimIR 默认 | Torch SDPA | Torch 显式组合 | MimIR 相对单线程 |
|---|---:|---:|---:|---:|
| 1 | 8.7074 | 4.4821 | 7.0833 | 1.00× |
| 4 | 6.4406 | 1.2587 | 4.8466 | 1.35× |
| 6 | 6.0842 | 0.8791 | 4.7632 | 1.43× |
| 8（6P+2E） | 7.5947 | 1.9567 | 4.1132 | 1.15× |

11个形状的单线程加速比几何平均（相同形状、相同后端自身对比）：

| 线程 | MimIR 默认 | Torch SDPA | Torch 显式组合 |
|---|---:|---:|---:|
| 1 | 1.00× | 1.00× | 1.00× |
| 4 | 1.31× | 2.86× | 1.82× |
| 6 | 1.39× | 3.52× | 2.09× |
| 8 | 1.23× | 2.18× | 1.58× |

本次修复解决内存生命周期，没有更换 tiling 参数或微内核。
默认路径仍物化 score/probability，softmax 未并行，因而多核扩展和 Torch CPU flash attention 仍有明显差距。
这由当前生成图的结构可以确认；本次未做分阶段性能剖析，不将差距定量归因到某一个阶段。
8线程混用6P+2E，结果不能解释为8个同质大核的扩展性。

旧版多线程默认路径使用受泄露上限约束的短批次，本次改为正常持续计时，
因此不将两次报告的延迟比直接解释为修复本身的性能收益。
实验性 online 路径仍无 OpenMP launch，完整四后端数据保留在 `summary.csv` 和各线程原始结果中。

## 配置与复现

- i5-14600KF；1/4/6线程分别用1/4/6个P核，mask为`0`、`0,2,4,6`、`0,2,4,6,8,10`；
  8线程加上E核`12,13`。主线程固定CPU0，工作线程实际mask与team大小均已审计。
- PyTorch 2.12.0+cpu，MKL2024.2，GNU OpenMP；MimIR 与 Torch 链接同一个 wheel 内的 `libgomp.so.1`。
- `-O3 -march=haswell -mno-avx512f`，Torch/MKL/oneDNN限制为AVX2；88个二进制均检查无ZMM/EVEX指令。
- `cpu-mr=6,nr=16,mc=320,nc=104512,kc=320,order=1`，`btensor:registers=128`。
- `OMP_PROC_BIND=TRUE`、`GOMP_CPU_AFFINITY`为上述mask，`OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`；
  `OMP_DYNAMIC=FALSE`、`MKL_DYNAMIC=FALSE`、`OMP_MAX_ACTIVE_LEVELS=1`。
- governor为powersave，正常turbo，未锁频。不同轮次可能有系统噪声；波动审计见 `timing-variation.json`。

```sh
PY=/workspaces/ml-compiler/gemm-bench/.venv/bin/python
ROOT=/tmp/mimir-attention-fixed
$PY bench/cpu_tiling/attention_cpu.py tools --root "$ROOT"
for t in 1 4 6 8; do
  $PY bench/cpu_tiling/attention_cpu.py prepare --root "$ROOT" --threads "$t"
  $PY bench/cpu_tiling/attention_cpu.py validate --root "$ROOT" --threads "$t"
done
$PY bench/cpu_tiling/attention_audit.py --root "$ROOT"
# 完成所有编译、校验和内存检查之后，串行计时：
for t in 1 4 6 8; do
  $PY bench/cpu_tiling/attention_cpu.py run --root "$ROOT" --threads "$t"
done
```

`compilation.json` / `preparation.json` 含每个内核的命令及IR/二进制SHA256；
`metadata.json` 含工具链、硬件、环境、编译器/插件哈希；`source-state.json` 标识准确修复源码。
编译器在提交前由相同源码构建，metadata中的当时HEAD与dirty路径据实保留，源码哈希和修复提交相对应。
`isa-audit.json`、`worker-audit.json`、原始9轮样本和回归日志均随本报告保存。

本次共 1584 个样本，最短连续计时 100.025 ms；所有默认路径样本中位净分配增长最大值为 0.0 bytes/调用。
