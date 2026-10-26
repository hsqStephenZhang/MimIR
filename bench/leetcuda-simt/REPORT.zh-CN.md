# LeetCUDA 操作的 MimIR SIMT 实现

## 本分支的提交边界

`feat/leetcuda-simt` 基于 GEMM PR 提交 `99d2dbf780c7eda5a2fb6ecd960e3386e27cd591`，仅收录 `bench/leetcuda-simt/` 下的用例、启动与校验工具、可选软件数学工具链和历史结果。**不包含 MimIR 编译器修复、GEMM 调优或相应 lit 回归。**

下文及 `results.json` 的实测数据来自带有本地编译器修改的隔离工作树，不是此分支原样构建的验证结果。完整重放依赖未随本次提交包含的 LLVM/NVPTX 优化、设备 bitcode 链接及相关 LLVM 表示修复；数学用例尤其依赖 `Xlink_llvm`/`Xopt` 在不使用 libdevice 时仍生效。默认 runner 不会自动应用这些修复，也不保证在未修复基底上全部通过。`--emit-only` 可以独立生成全部用例；执行时可用 `--mim` 指定兼容的外部构建。

下文关于编译器修复、lit 回归、GEMM 报告和 `artifacts/` 日志的描述保留为此前实验记录；被引用的分支外文件及忽略产物未随本次提交提供。`results.json` 中的路径与源码 hash 同样记录当时的本地环境，并非本分支编译器源码一致性的声明。

## 范围与入口

参考 `/workspaces/ml-compiler/LeetCUDA` 的操作定义和算法分类，独立实现 Mim GPU kernel；没有复制其 GPLv3 CUDA 实现。实验在 `/workspaces/compile/MimIR/.worktrees/pr2-gemm-bench` 完成，未修改主工作区；本分支仅提交上述用例目录。

当前完成 **72 个编译、GPU 执行和数值校验用例**，包括不同尺寸、dtype 和算法变体，不等于 72 个独立算法，更不代表复现了 LeetCUDA 的全部优化变体。按要求撤销了 pow/erf 的 libdevice 映射，改用优化后的 LLVM 标准 intrinsic lowering 和软件数学运行时；新增 F64 pow、erf、erfc 的实际 GPU 验证。

| 文件 | 内容 |
| --- | --- |
| `simt.mim` | 标量数学、类型化输入输出、循环和 GPU map 启动器 |
| `block.mim` | shared-memory tree reduction、同步和 block 启动器 |
| `run.py` | 每个 kernel 的 Mim 函数体、尺寸、dtype、源码生成及逐项执行 |
| `check.cu` | CUDA Driver 启动器、独立 CPU 数值参考及 CUDA-event 计时；不包含计算用 CUDA kernel |
| `results.json` | 72 个用例的实测时间、误差、寄存器/shared 用量、校验状态及 SHA256 |
| `toolchain/` | 固定版本 LLVM libc/OpenLibm 的构建、late LLVM pass、F64 pow 补丁、独立复现及方案说明 |
| `artifacts/<name>/<name>.mim` | 生成的完整、自包含 Mim 源码，可以单独编译 |
| `artifacts/<name>/` | LLVM、PTX、cubin、编译日志、校验日志和 sanitizer 日志；此目录不纳入 Git |

`--emit-only` 可以在不运行 CUDA 工具的情况下重新生成完整源码。操作计算在 Mim kernel 内完成，不是调用 CUDA/C++ 实现的包装。

## 已实现覆盖

| LeetCUDA 目录/操作 | 本轮实现 | 限定 |
| --- | --- | --- |
| `elementwise` | Add、Mul，FP32 和 FP16 | FP16 存储，显式 FP32 运算，FP16 输出；非 float4/half2 |
| `relu`、`sigmoid`、`swish`、`elu`、`hardshrink`、`hardswish` | FP32、FP16 输入输出 | ELU α=1，Hardshrink 阈值 0.5；严格阈值两侧有测试 |
| `gelu` | tanh、erf 两种公式，均支持 FP32/FP16 存储 | 没有把 tanh 近似冒充 erf |
| `embedding` | gather，FP32 数据；包括原生 I32 索引版本 | 仅合法索引；另保留早期浮点索引适配用例 |
| `mat-transpose` | 直接转置、16×16 shared 转置，stride=17 | 37×65、1031×2053，均覆盖边缘 tile |
| `reduce`、`dot-product` | Sum、Max、Dot；串行部分归约及 256-thread block tree | strided partials 支持归约宽度超过 256；另有 FP16 输入/FP32 累加 Dot |
| `sgemv`、`hgemv` | FP32 串行及 block GEMV；FP16 输入/FP32 累加 GEMV | FP16 版本不是 half2，也不是 FP16 累加 |
| `sgemm` / SIMT half GEMM | FP32、FP16 输入/FP32 累加的朴素矩阵乘 | 33×17 · 17×35；此前 PR 的 shared/register-tiled GEMM 保留，见 `../gemm-pr2/REPORT.zh-CN.md` |
| `softmax` | 普通、减最大值、online recurrence、block safe、block LogSoftmax | 稳定版本另外测试整体 +1000 的输入；普通版本不承诺抗溢出 |
| `layer-norm` | 两遍统计、Welford、block tree；FP16 输入/FP32 输出变体 | 每行 129 个值，ε=1e-5；γ=1、β=0，没有实现任意 affine 参数 |
| `rms-norm` | 直接归约、block tree、FP16 输入/FP32 输出 | 每行 129 个值，ε=1e-5，scale=1 |
| `rope` | 相邻偶奇元素配对旋转，FP32 和 FP16 输入/FP32 输出 | 31×34，θ=10000，position=行号；不是 split-half 配对约定 |
| `histogram` | bin-owned 浮点适配及原生 I32 输入/I32 计数 | 4099 个输入、17 个 bin；O(input×bins)，不是 atomic histogram |
| `nms` | 一个 block 内并行 per-box IoU mask，加顺序 greedy 消解 | 已按分数排序的输入，N≤64 的设计，本例 N=37；返回 keep flags，不做排序/索引压缩 |
| attention / `flash-attn` 数学路径 | stable 两遍、online、causal online 的 FP32 SIMT attention | S=17、D=9；一线程一个输出，重算 dot；不是高性能 FlashAttention |
| 分块 attention 合并 | log-sum-exp 加权合并两个 partial outputs | 验证稳定加权公式，不含多 GPU 通信 |

`add_n1`、`add_n256`、`add_n257` 专门验证小尺寸、整 block 和尾部；常规逐元素长度 4099。归约覆盖 4099、513、259 等非整除宽度。NMS 输入包含非传递抑制链、恰好 IoU=0.5、完全不重叠和全部重叠，检查 strict `IoU > 0.5`。

本轮触及上述 21 个主要 SIMT 操作目录；`interview`/`transformer` 的完整模型与教学项目、CUTLASS/Triton 程序及架构专用目录，不算作已移植项目。

## 实际验证

设备：GTX 1650 SUPER，sm_75，driver 595.71.05；CUDA 13.2。GPU 工作顺序执行，没有并发跑 bench/sanitizer。

- 72/72 编译、真实 cubin 执行通过；三组输入逐元素对照 CPU 参考，共 **13,095,201** 个输出比较。
- FP16 输入在 CPU 参考前先按真实 half 舍入；浮点容差 `3e-4 + 5e-4 × |reference|`，I32 计数要求完全一致。
- 独立 `math_pow`、`math_erf` 用例覆盖负底数、NaN、Inf、signed zero、次正规数、上溢和下溢；被测有限值与舍入到 FP32 的 double 参考相差不超过 2 ULP，检查零的符号及非有限值类别。
- F64 pow/erf/erfc 各 8,193 个输出、三组输入，共 **73,737** 个比较；真实 double 存储和计算，以更高精度的 long double CPU 结果舍入到 double 为参考，同样要求 ≤2 ULP。当前 host 的 long double 有 64 位有效尾数，double 为 53 位；不支持更宽 long double 的 host 会显式报错。
- 全部 72 个用例通过 memcheck。
- 12 个 shared-memory/NMS 用例通过 racecheck 和 synccheck，均无 hazard/error。
- 编译器修复后完整 lit：430 通过、5 个平台/功能不支持而跳过；26/26 单元测试通过。新增 affine、SEO 和实际 GPU 回归，详见下节。
- `lit/nvptx/math_intrinsics.mim` 检查 FP32/F64 pow 保持 LLVM intrinsic，erf/erfc 保持普通函数调用，没有 NV vendor 映射。该 lit 测试只验证 IR；F64 运行证据来自上述三个独立 GPU 用例。

除独立数学测试外，操作用例输入是有限值，没有声称其他操作覆盖 NaN、Inf，也未覆盖空输入、非法索引或任意动态尺寸。当前程序按 `Case` 的常量尺寸生成专用 kernel。block normalization/softmax 版本当前只实现被测的 129 列路径，不是任意宽度的通用库接口。

### F64 实测

以下每个 kernel 都处理 8,193 个 double，最大 ULP 误差取三组输入，时间取最终完整轮次的最后一组输入；不同函数输入分布不同，不作吞吐量横向排名。

| Kernel | 输出比较数 | 最大误差 | Kernel 时间 |
| --- | ---: | ---: | ---: |
| `math_pow_f64` | 24,579 | 1 ULP | 11.604 µs |
| `math_erf_f64` | 24,579 | 1 ULP | 23.673 µs |
| `math_erfc_f64` | 24,579 | 2 ULP | 22.606 µs |

检查了生成 IR 的 double load/store 和 double 数学调用，没有 F64→F32 截断；三个 kernel 的 PTX 和软件运行时均没有 `__nv_*` 引用。测试覆盖整个二进制指数范围的采样、erf/erfc 分段边界的相邻浮点值及特殊值，但不是穷举或 MPFR 误差证明。

## 每个版本的 bench

每个用例均在通过数值校验后计时：20 次 warmup、5 组×100 次 CUDA-event launch，取每次 kernel 时间的组中位数。计时只含 device kernel，不含分配、拷贝、编译和 CPU 参考；使用最后一组输入。以下为前一轮 69 用例的对比，单位 µs；加入 F64 后的重跑数据见 `results.json`。

| 同一计算任务 | 功能基线 | 协作/online 版 | 基线时间 / 新版时间 |
| --- | ---: | ---: | ---: |
| 19×513 row sum | 30.249 | 2.046 | 14.78× |
| 19×513 row max | 25.964 | 2.014 | 12.89× |
| Dot，4099 | 42.108 | 2.516 | 16.74× |
| GEMV，37×259 | 6.267 | 2.300 | 2.72× |
| Safe Softmax，7×129 | 14.793 | 2.706 | 5.47× |
| LayerNorm，7×129 | 8.564 | 2.794 | 3.07× |
| RMSNorm，7×129 | 5.778 | 1.866 | 3.10× |
| Transpose，37×65 | 1.309 | 1.420 | 0.92× |
| Transpose，1031×2053 | 185.467 | 137.933 | 1.34× |
| Attention，两遍 → online | 6.723 | 4.567 | 1.47× |

shared transpose 在小尺寸没有稳定收益，保留为教学/协作版本，不默认宣称更快。微秒级小 kernel 受 launch/时钟噪声影响；这些数字只比较本轮 Mim 基线和变体，不是与 LeetCUDA、PyTorch 或 cuBLAS 的性能排名。此前 GEMM/cuBLAS 对比仍以独立 GEMM 报告为准。

改用软件数学运行时后：GELU-erf FP32 2.724 µs、FP16 存储版 2.876 µs；RoPE FP32 4.157 µs、FP16 输入版 4.764 µs。没有冻结旧 libdevice 方案的独立同轮样本，因此不声称这是相对旧方案的性能提升。

## 遇到的问题与处置

| 问题 | 证据/现象 | 处置 |
| --- | --- | --- |
| GPU 没有普通 erf 数学函数定义 | GELU-erf：`ptxas fatal: Unresolved extern function 'erff'` | 撤销此前的 `__nv_erff` 映射；保留普通 `erff` 调用，链接 LLVM libc 的通用 FP32 实现 |
| LLVM NVPTX 无法合法化 pow intrinsic | 独立、通过 verifier 的 `pow.ll` 在系统 LLVM 22.1.3 上仍报 `Cannot select ... fpow`，排除了 Mim 前端 | 撤销此前的 `__nv_powf` 映射；Mim 输出保持 `llvm.pow.f32`，O2 优化后通过 LLVM `IntrinsicLowering` 转成 `powf`，由 LLVM libc 实现 |
| LLVM libc F64 pow 次正规数输入错误 | `pow(0x0.2100000000000p-1022, -0.875)` 得到约 `2.02051e286`，参考约 `9.43157e269` | 对构建副本应用明确补丁：归一化后从缩放值重新提取 exponent，再减 64；原始 dependency checkout 不改 |
| LLVM libc F64 pow 中间溢出 | `pow(0x1p512, -2)` 的快捷分支计算 `1/(x*x)` 得到 0，正确结果为可表示的 `0x0.4p-1022` | 负二次幂改走原有通用 pow 算法；保持两 ULP 阈值，不用放宽容差掩盖错误 |
| LLVM libc 缺少通用 F64 erf | 固定版本中有 `erf.h`，但没有 generic double 实现 | 固定版本 OpenLibm 提供 erf/erfc，LLVM libc 提供所需 exp/fabs；禁用 NVPTX 不支持的 host weak aliases，不修改数学公式 |
| shared `%mem.slot` 遇到 SEO 崩溃 | GDB 确认分析当前 binder 为 mutable Pack，却将 `curr_mut<Lam>()` 的空值传入 `mk_phi` | 已修复：仅在真实 lambda 前驱上下文传播 slot phi；不是 shared 地址空间或泛型 `Def` 构造器的问题。最小用例原先 SIGSEGV 139，现在通过，原始 GPU reduction 亦能直接用 slot |
| affine 循环的 LLVM 溢出标记 | 无符号循环使用 `add nuw nsw i3`；`Idx 7` 的 3→4 跨 signed 范围，LLVM O2 后程序超时；I8/I32 同样存在 | 已修复：`LowerFor` 的 increment 改为 `Mode::nuw`，与无符号循环比较一致，不再错误承诺 signed no-wrap；新增跨 signed 边界的 CPU/GPU 回归 |
| 把分支 continuation 当作返回值传递 | `cannot convert the type of a basic block: Cn [mem.M 1, mem.M 3]` | helper 的分支捕获外层 return，不再把 basic-block continuation 当作普通运行时值 |
| 泛型 CPS callee 的隐式参数 | `ret` 报 callee 不是 returning continuation | 显式实例化 `fold @F32`、`fold @I32`、`run @(Ta,Tb,Tc)`，无需修改语言 |
| FP16 存储与 FP32 oracle 不一致 | 未先量化输入会把存储舍入混入实现误差 | harness 先按 half 实际舍入，再计算 double 参考，并按真实 dtype 分配/拷贝 |
| 过滤执行丢失已有进度 | 初版 manifest 每次被选中用例覆盖 | 已按名称合并并逐项落盘；sanitizer 证据只在生成源码、harness 和 cubin 的 hash 都一致时保留 |
| 编译器被信号终止时日志空白 | shared slot 的 compile log 最初为空 | runner 记录 return code，并把终止信号写入日志，失败返回非零 |

`src/mim/plug/ll_nvptx/phase/ll_nvptx.cpp` 中新增的 pow/erf 分支已全部撤销。后端只扩展 `Xlink_llvm`，允许在不需要 libdevice 时链接普通设备 bitcode；数学实现和 late lowering 放在独立、显式启用的 `toolchain/`。未改变 GELU/RoPE 公式，也没有用简单 `exp(y*log(x))` 替换通用 pow。独立 FP32/F64 数学 kernel 和软件运行时均无 `__nv_*` 引用；RoPE 保留原有 sin/cos 路径，但没有 `__nv_pow*`/`__nv_erf*` 调用。详见 `toolchain/README.md`。

### MimIR 编译器修复与逐项 bench

两个修复分别位于 `src/mim/plug/affine/phase/lower_for.cpp` 和 `src/mim/plug/mem/phase/seo.cpp`。SEO 的分析会遍历所有 mutable binders，并非只遍历控制流 lambda；Pack 不是 slot phi 的控制流前驱，不能给它构造以空 lambda 为 operand 的 phi。修复在分析边界区分该上下文，没有给底层 IR 构造添加空值兜底。

| 新增回归 | 验证内容 |
| --- | --- |
| `lit/affine/for_unsigned.mim` | `Idx 7`、`Idx 8`、`Idx 256`、跨 `INT_MAX` 的 I32；检查 IR 无错误 `nsw`，并执行 O2 产物。修复前产物运行超时 |
| `lit/mem/seo/shared_slot.mim` | shared slot、捕获它的 callbacks、mutable Pack 共同触发原路径；实测去掉修复后 SIGSEGV 139 |
| `lit/nvptx/shared_slot.mim` | 256 个 GPU threads 通过真实 shared slot 交换数据，barrier 后用 `Idx 7` 循环求邻居和，包含跨 warp 和数组环绕读取；CPU 检查全部 256 个输出 |

实际 GPU 联合回归通过 memcheck、racecheck、synccheck。原来崩溃的完整 19×513 row reduction 也已从 launch-shared 改为 kernel 内 `%mem.slot` 的独立复现版本：三组输入共 57 个比较通过，三项 sanitizer 均通过，17 个寄存器、1024 bytes shared，实测 **2.285 µs**。源码、命令、SHA256、数值结果及日志保留在 `artifacts/compiler-repairs/`，数值汇总为 `shared-slot-results.json`。

按每项修复后跑 bench 的要求，用相同的三个原有协作 kernel 依次测量，单位 µs：

| Kernel | 两项修复前 | affine 修复后 | SEO 修复后 |
| --- | ---: | ---: | ---: |
| `reduce_rows_block` | 2.283 | 2.227 | 2.283 |
| `softmax_safe_block` | 2.249 | 2.701 | 2.375 |
| `layer_norm_block` | 2.380 | 2.990 | 2.656 |

三个阶段每个 kernel 的 cubin SHA256 均完全一致；这些微秒级时延波动不能归因于修复，也不能据此声称加速。此次是正确性修复，没有改变这三个已有 I64/launch-shared kernel 的机器码。阶段快照保存在 `artifacts/compiler-{before,idx,seo}-results.json`；`results.json` 则是修复后的完整 72 用例重跑，并刷新全部 memcheck 和 12 个协作用例的 racecheck/synccheck 证据。

额外用修复后的编译器重跑原 explicit-FMA GEMM 的 **511³**：9 轮中位数 **0.133219 ms**，同轮 cuBLAS **0.137050 ms**，耗时低约 **2.8%**，数值校验通过。这是当前同轮比较，不是相对于修复前编译器的加速证明。完整源码、原始 trial、资源与汇总 CSV 保存在 `artifacts/compiler-repairs/gemm/`，未覆盖此前 GEMM 的历史阶段。

为保留对比，原有 72 用例仍用原来的 I64 与 launch-shared 写法；新增回归直接验证不依赖这两项绕过的路径。此次移除错误的 signed no-wrap 承诺，保留既有 `nuw` 契约，不代表新增了任意无符号环绕或负步长循环语义。

## 记录并跳过的变体

| 类别 | 当前限制 |
| --- | --- |
| warp shuffle / ballot 归约、原版 NMS mask | 此工作树 GPU 插件没有这些原语；本轮使用 shared tree 和 per-thread bitmask 替代，没有实现对应 warp 指令版本 |
| atomic histogram / 跨 block atomic 归约 | GPU 插件没有 atomic add 接口；保留整数 bin-owned 基线，不冒充原版原子算法 |
| float4/half2/FP16x8 显式向量化 | 本轮未实现打包算术和向量访存；NVPTX emitter 当前转换类型时关闭 SIMD。FP16 版本只代表真实 half 存储，不代表 half2 性能 |
| cp.async / 多级异步流水线 | sm_75 无 sm_80+ cp.async；本轮未扩展异步拷贝原语，也未重做此前失败的 GEMM prefetch |
| Tensor Core / TF32 / warp-specialized MMA / TMA | 当前设备不能验证相应硬件路径；未用普通 SIMT 数字冒充这些版本 |
| 完整 FlashAttention、高性能 half GEMM、完整 Transformer | 已有数学正确的 SIMT attention/half GEMM 基线，但没有复现 tile/warp/pipeline 全套调度和模型集成 |
| 通用 NMS | 当前仅 sorted boxes、N≤64、keep flags；一般排序、任意 N、多字 mask 和压缩返回未实现 |
| 通用归一化/大型多阶段归约 | 未实现任意宽度 block normalization、任意 γ/β、跨多个 kernel 的大规模归约调度 |
| 其他精度的软件数学运行时 | 当前覆盖 FP32 powf/erff、F64 pow/erf/erfc；尚未提供 FP32 erfc，也未穷举所有输入或使用 MPFR 做严格误差证明 |

优先扩展方向是带 active-mask 契约的 shuffle/ballot，以及明确内存序/作用域的 atomic；这些会同时帮助归约、softmax、normalization、histogram 和 NMS。当前已完成的标量/共享内存 SIMT 版本不依赖这些新原语。

## 重放

在隔离工作树根目录执行，使用已经构建的 `build-clang`：

```bash
python3 bench/leetcuda-simt/toolchain/build.py
python3 bench/leetcuda-simt/run.py
python3 bench/leetcuda-simt/run.py --filter '^softmax_safe_block$'
python3 bench/leetcuda-simt/run.py --filter '^math_.*_f64$' --sanitize memcheck
python3 bench/leetcuda-simt/run.py --sanitize memcheck
python3 bench/leetcuda-simt/run.py --filter '(block|shared|nms_presorted)' --sanitize racecheck
python3 bench/leetcuda-simt/run.py --filter '(block|shared|nms_presorted)' --sanitize synccheck
python3 bench/leetcuda-simt/run.py --filter '^attention_online$' --emit-only
cmake --build build-clang --target mim_lit_tests
./lit/lit build-clang/lit -j1 --filter '(affine/for_unsigned|mem/seo/shared_slot|nvptx/shared_slot)\.mim$'
```

也可用 `--mim` 指定同一插件布局的编译器，`--out` 指定生成目录，或 `--math-runtime` 指定已构建的软件数学目录。`--no-software-math` 用于诊断原始 LLVM 路径，不会回退到 vendor pow/erf。目标硬件/工具参数目前固定 sm_75。一个用例失败会记入 manifest，并继续其他用例，最后返回非零；过滤表达式没有匹配项也会报错。
