# ss420e GPU 内存检查

编译器源码为 `08e40d6fc4`。GPU 0：RTX 3090 / sm_86，CUDA 12.1。

默认 GPU tiling 的 fold 使用固定的累加器、私有临时存储和 work-group shared slots，
没有经过此前泄漏的 reference Tensor fold 提前展开路径。shared slots 随 CTA 结束回收；
Tensor Core 累加器保留为 fragments。并非所有 GPU 临时量都已变成寄存器：下面的 K 尾块
仍包含一次 device `malloc(64)` 及匹配的 `free`，因此同时做了运行检查。

| 当前编译的路径 | 形状 M×K×N | 重复启动 | 显存增长 | Compute Sanitizer |
|---|---|---:|---:|---|
| 3-stage pipeline | 16×16×16 | 1000 | 0 B | 0 leaks / 0 errors |
| peeled K tail | 32×24×32 | 1000 | 0 B | 0 leaks / 0 errors |
| Tensor Core | 16×16×16 | 1000 | 0 B | 0 leaks / 0 errors |

每组先验证完整输出，再预热 20 次，在同步后读取 `cuMemGetInfo`，重复启动 1000 次，
同步后再次测量并校验输出。使用 CUDA Driver API 直接管理 buffers、module 和 primary
context，避免把 PyTorch caching allocator 的保留内存当作泄漏。退出前显式释放全部资源。
三个原始 `memcheck.log` 均包含 `LEAK SUMMARY: 0 bytes leaked in 0 allocations`。
本检查覆盖上述 kernel 路径，不是对所有 GPU 算子及完整生成 host wrapper 的无泄漏证明。

另发现旧 benchmark driver 的独立问题：五个端到端 driver 覆盖返回的 malloc-owned 输出
而未释放前一个结果；`kbench` 未释放 device buffers、CUDA events、module 和 primary
context。本次补齐正常退出的清理，并在找不到正确参数顺序的分支释放已创建的资源。
这些修改不影响 gemm-bench 的预分配 CUDA kernel 计时。

同一 16³ fixture 的旧 `kbench` 实测泄漏 3 块 device buffer、合计 3072 B，
见 `kbench-before-memcheck.log`。`kbench-memcheck.log` 为修复后的真实 CUDA 运行结果，
泄漏降为 0 B / 0 allocations，错误数也为 0。
五个 host-return driver 通过 C++ 编译语法检查；此处没有运行它们的完整端到端 wrapper。

复现：`prepare.py` 记录本次生成 PTX 的命令和本地目录；`manifest.json` 记录源码身份、
PTX SHA256、launch 参数以及 kernel 的 device heap 分配/释放站点数。
在 ss420e 用 `/usr/local/cuda/bin/ptxas -arch=sm_86 kernel.ptx -o kernel.cubin` 汇编后，运行：

```sh
CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  /home/sunqi/zc/mimir-bench/gemm-bench-gpu-env/bin/python check.py /absolute/path/to/case
```

K 尾块的 kernel 捕获两个向量视图和三个原始 buffers；`check.py` 的五参数映射只针对
此次 fixture。其余两个 fixture 通过数值校验确定三个参数的顺序。
远端目录：`~/zc/mimir-bench/gpu-leak-check-20261112`。
