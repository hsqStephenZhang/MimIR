# tritonbench 算子上的 multi-level tiling（H200 NVL，2026-11-13）

`tb.py <op> key=value…` 把 `<op>.mim` 模板（`@M@` 等占位符）实例化，编译到 nvptx，用 PyTorch 生成输入并校验结果，
用 nsys 统计每次调用的 kernel 时间；`tb_driver.cpp` 是通用驱动。用 tritonbench 的 venv 运行：

```
~/workspaces/tritonbench/.venv/bin/python tb.py softmax M=4096 N=4096 R=1 I=8 P=0
```

驱动连续调用 5 次，输入可留在 L2；表中 torch.compile 用同样“不清 L2、连续重复”的口径测（CUDA event，200 次平均），
括号里是 tritonbench `do_bench`（每次清 L2）的值。单位 ms。

## 访存密集型

| 算子 | 形状 | 调度 | 修复前 | 修复后 | torch.compile |
|---|---|---|---:|---:|---:|
| softmax | 4096×4096 F16 | 一行一个 work-group，item 8 元素 | 0.297（item 32） / 0.106 | **0.0405** | 0.0426（0.0445） |
| rms_norm | 2048×4096 F16 | 同上 | 0.0317（item 4） | **0.0102** | 0.0131（0.0159） |
| layer_norm | 4096×2048 F16 | 同上 | 0.0279（item 4） | **0.0104** | 0.0168（0.0194） |
| swiglu | 8192×4096 F16 | 4×1024 work-group，item 8 元素 | 0.104（item 4） | **0.0474** | 0.0513（0.0550） |

## GEMM + epilogue

| 算子 | (M, N, K) | 调度 | MimIR | aten | inductor / pt2 Triton |
|---|---|---|---:|---:|---:|
| addmm（bias，F16 输出） | (20120, 512, 1536) | 128×128，`wg_red_tma` 3 级，cap 尾块 | 0.0805 | 0.0608 | 0.0772 |
| addmm | (19735, 512, 1536) | 同上 | 0.0798 | 0.0592 | 0.0770 |
| linear + gelu(tanh) | (8192, 4096, 1024) | 128×128，`wg_red_tma` 3 级 | 0.166 | 0.160 | 0.163 |
| linear + gelu(tanh) | (8192, 8192, 2048) | 同上 | 0.552 | 0.529 | 0.535 |

纯 GEMM（F32 输出）同时受益于成对写回：4096³ 0.2545 → 0.225，8192³ 2.18 → 2.02（cuBLAS 0.204 / 1.82，约 90%）。

## 发现并修复的不足

1. **wgmma 片段不支持非恒等 epilogue。**
   `bias + a·b`、转 F16、gelu 落到通用 `lower_map_reduce`，类型错误；报错位置还指向 `gpu.mim` 的无关声明。
   现在 `store_epilogue` 把逐点 `map_reduce_post` 作用在片段上，其它输入沿写回的内存链逐元素读取，epilogue 本身是纯函数。
   - 第一次实现把读放在每元素的 lam 里，`fresh_mem` 让整个 fold 被复制了 256 份；
   - bias 是 region 输入的 `slice_box` 视图，每个线程 malloc 一份拷贝（`STACK:512`）；改为用 `panel::tensor_read` 穿过视图直接读源张量。
2. **片段逐元素 16 位写回。** 同一行相邻两列成对写（F16 为 `st.global.b32`，F32 为 64 位），addmm 0.100 → 0.081。
3. **绑定层不支持尾块。** M=20120 这类真实形状报 “a bound tensor.tiled level requires literal extents”。
   region 用 TMA 填充时，work-group 的最后一个 tile 按整块计算（TMA 越界补零），写回按 `origin + row < M` 跳过越界元素。
   分支的参数里带上要写的值，否则计算这对值的 fold 会被复制进每个分支（动态 shared 一度变成 3 MB）。
4. **逐元素路径的全局访存是 16 位标量。** 四个原因叠加，缺一不可：
   - 仿射映射的坐标收窄用 `conv.u`（`trunc`），`zext(trunc x)` 让 LLVM 证明不了相邻元素连续。
     映射输出按语义落在 `sout` 内，改为 nuw 的 `core.idx`，ll 后端生成 `trunc nuw`。
   - kernel 参数是一个指针数组，LLVM 不知道它们互不重叠，不敢把后面的读提到写之前。
     每个捕获的指针都是 outline 单独分配的设备内存，DeviceEmitter 给追溯到不同参数的访存加上不同的 `!alias.scope` / `!noalias`。
   - 参数指针的对齐未知。`llvm.assume` 的 align 操作数包会在指针下沉时被丢掉（InstCombine 还会把布尔形式改写成操作数包），
     所以改为在转成通用地址时插 `llvm.ptrmask(p, -16)`，对齐成为值本身的属性。
   - libdevice 函数（`__nv_expf` 等）的声明没有内存属性，调用把访存链切断；另外 `nvvm-reflect` 排在 O2 之后、O2 的展开阈值偏小。
     现在声明为 `nounwind willreturn memory(none)`，device 流水线改为 `nvvm-reflect,default<O3>`。
5. **行归约的跨 item 合并逐层过 shared memory。** 部分结果由 F32 组成、item 数为 2 的幂且是 32 的倍数时，
   先在 warp 内按分量做 5 轮 `shfl_xor`，shared 树只合并每个 warp 的一个结果（softmax 12 → 6 次 barrier）。

## 还没解决

- 不同秩输入的 `map_reduce`（如 `(x: 2 维, ml: 1 维)`）推不出类型，要显式写 `@((H, ML), (2, 1), ((M, N), M))`。
- 调度对每个 item 的元素数很敏感（softmax item 32：0.297，item 8：0.0405），目前完全手选，没有代价模型。
- softmax 两遍都读全局的 x（L2 命中掩盖了代价）；行常驻寄存器还表达不了。
- cross_entropy 需要按标签取 `x[i, t[i]]`，仿射的 `proj_map` 表达不了数据相关的下标。
- 尾块的写回屏蔽和 TMA 越界补零是 lowering 的约定，相位类型里没有建模，`check_sync` 也不检查。
