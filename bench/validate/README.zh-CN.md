# 调度的差分验证（D3 第 1 级，2026-11-13）

`fuzz.py <op> [cases] [seed]` 随机采样形状与调度参数，把每个用例编译两次，在相同输入上比较：

- **参考**：把 `tile_req (op, …)` 换成 `op`（它的语义），在 CPU 上编译运行；
- **被测**：`-X tensor:tile-fuse=1` 编到 GPU（`MIM_FLAGS` 追加编译选项，如 `-X tensor:pipeline=1`）。

输入是小整数，收缩的每个和与积在输入与累加器类型里都精确，任意重结合必须逐位一致；含超越函数的算子按容差比较。
结果分五类：`pass`、`fail`（结果不一致）、`rule`（elaboration 时被 `tensor.check` 规则拒绝）、`late`（lowering 才拒绝，说明规则还没写出这条限制）、`error`（其它）。

`one.py <op> '<json params>'` 跑单个用例。`gemm.mim` 是不带 epilogue 的 GEMM 模板；`addmm`、`softmax`、`rms_norm` 用 `bench/tritonbench` 的模板。

## 本轮发现

| 现象 | 原因 | 处理 |
|---|---|---|
| `wg_red`（非 TMA）的 wgmma GEMM 结果错误，pipeline=1 或 k 只有一块时 | 面板用普通存储填进 shared，wgmma 经 async proxy 读，未流水的路径缺 `fence.proxy.async` | `LowerToMem::publish_panels`：没有流水 ring 发布面板时，wgmma 前补 fence 与屏障；`lit/nvptx/tile_wgmma_unpipelined.mim` 在 GPU 上执行 |
| `wg_red_tma` 在 k 只有一块时结果错误 | 块数 1 时流水深度为 1，退回上面那条路径 | 同上 |
| `wg_red_ws` 在 k 只有一块时非法指令（CUresult 715） | specialized 结构在绑定时已确定，单块没有 ring 可填 | 新规则：specialized 归约至少两个块（`lit/error/tile_ws_blocks.mim`） |
| 不开 `tile-fuse` 时，带层级的程序报 “cannot coerce to `TileLevels_…`” | `tile_req` 活到 RWPhase 重建时，依赖 sigma 的分量类型在新 world 里重新实例化的递归 nom 与值的类型副本结构比较失败；类型改动之前（4f6527ca3）就存在 | 未修；参考程序改为源码层去掉 `tile_req` |

lit 里 `gpu/` 下的 wgmma 测试只在不加载 gpu 插件时执行（host 循环），`nvptx/` 下的在 GPU 上执行但都带 `-X tensor:pipeline=2`；
tritonbench 只用 TMA/特化且 K 很大。所以前两条一直没被测到。

`late` 中最常见的是 “a wgmma operand panel needs its swizzled layout”：非 TMA 路径下 64 行的 work-group tile，以及 tile 大于矩阵的情况。
它们只拒绝、不算错，留作规则或 `panel_layout` 的后续工作。
