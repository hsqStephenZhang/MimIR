# 规则即约束的调度搜索（D4 第一步，2026-11-13）

`tune.py <op> key=v1,v2,… …` 枚举 tritonbench 模板的调度参数组合：

1. 每个组合先渲染并只做 elaboration（`mim -p gpu x.mim`，约 0.25 s）；`tensor.check` 等规则拒绝的组合不编译、不运行。
2. 规则合法、但一个 work-group 超过 1024 个 work-item 的组合按代价剪掉：`gpu.outline` 让线程分轮次执行，正确但远离快路径，而且在大 grid 下会崩溃（见 `bench/validate/README.zh-CN.md`）。
3. 其余组合交给 `tb.py`：与 PyTorch 比对结果，并用 nsys 计时；测出 0 个 kernel 的算失败。

没有代价模型：剪枝全部来自规则，排序来自实测。

## 结果（H200 NVL）

| 算子 | 空间 | 规则拒绝 | 代价剪枝 | 合法且正确 | 最优 | 原手选 | 合法调度间的差距 |
|---|---:|---:|---:|---:|---|---|---:|
| softmax 4096² | 40 | 0 | 12 | 28/28 | R=1 I=16 P=0：**0.0314 ms** | R=1 I=8 P=0：0.0385 ms | 14.8× |
| addmm (20120, 512, 1536) | 108 | 9 | 0 | 81/99 | TM=128 TN=128 S=3 tma：0.0798 ms | 同左 | 3.3× |

softmax 比手选快 1.23×，也快于 torch.compile（0.043–0.055 ms）；addmm 的手选已是空间内最优。
addmm 中 18 个合法组合在 lowering 晚期被拒（TM=64 的非 TMA 面板没有 swizzle 布局），没有算错的。
