# MimIR / portmatching / GHL 对照

2026-11-02，在本地三个 checkout 上运行。此次只增加测试、适配器和测量工具，未改匹配器实现。

后续阶段计时、分配统计和采样见 [搜索开销排查](PROFILE.zh-CN.md)。

在下面这批固定端口 DAG 上，当前 MimIR 的搜索耗时高于 portmatching 和 GHL；MimIR 原生多根用例中，Indexed 相对 Exhaustive 仍有明显收益。两项结论适用的工作负载不同。

## 语义与机制

| 项目 | 本次调用的接口 | 搜索机制与边界 |
| --- | --- | --- |
| MimIR | `GraphPattern` / `find_matches` | 共享 capture、trail 回溯、结构预过滤、绑定驱动的反向索引和候选选择；还生成供安全重写使用的读取集合。依赖类型约束可进入同一绑定环境；本次共同测试未使用类型约束。 |
| portmatching | `PGNaiveManyPatternMatcher` / `PGManyPatternMatcher` | 前者逐模式搜索，后者编译共享约束自动机；利用固定端口路径和作用域内的绑定进行搜索。此处采用具体的无权 `PortGraph` 接口。 |
| GHL | `Isomorphism::discover`，分别选 VF2 / VF3 | 一般图的单射子图同构、顶点/边比较器和结果校验回调。调用实际 GHL 包装层并收集其映射；VF3 使用仓库提供的 Boost 补丁和默认顶点分类。 |

MimIR 不要求不同 capture 必须绑定不同节点，因此适配器把每个模式顶点声明为 distinct anchor，补上端口图/同构所需的顶点单射约束。相同 capture 则保持节点身份一致。

portmatching 与本适配器采用正边约束：模式中的边必须存在，开放端口不施加额外条件。GHL VF2/VF3 采用 induced matching：已选顶点之间多出一条边也会拒绝。共同 benchmark 的 oracle 会验证所有预期映射均无额外内部边，因此这里可以比较完整结果集合。

这一差异另有实际运行验证：三节点链的目标图增加首尾连边后，MimIR 和 portmatching 各返回一个匹配，GHL VF2/VF3 返回零个，见 [semantic-boundaries.csv](results/2026-11-02/semantic-boundaries.csv)。此案例不混入共同性能表。

## 移植测试

[pattern_portgraph.cpp](../../test/pattern_portgraph.cpp) 共六组测试：

1. 原样移植 portmatching `src/concrete/portgraph.rs::single_pattern_match_complex` 的拓扑：100 个顶点、4 条模式边、3 个重叠嵌入。核对完整有序顶点映射，并覆盖候选正/反序及 Indexed/Exhaustive。
2. 修改源/目标端口以及拆开共享顶点，检查拒绝匹配。
3. 检查不同顶点不能折叠为同一节点，开放端口允许连接模式以外的节点。
4. 比较节点约束分开描述的长链，检查长度 8 的模式在默认深度预算下成功。
5. 记录 GHL induced 语义与 MimIR 正边约束的差异。
6. 适配 GHL `tests/cpp/test_{vf3_,}isomorphisms.cpp::InsertVertexAfterID` 的匹配选择部分：顶点谓词、拒绝已处理节点的校验、选取第一个有效结果。没有移植其可变图插入操作。

新增测试在既有全套测试中运行通过：`mim-test` **210/210，4719 个断言**；`mim-tensor-pattern-test` **16/16，143 个断言**。外部库运行的是本目录的共同 corpus，不表示已运行它们各自的全部测试。

适配器只接受按拓扑顺序编号的三输入/三输出端口 DAG。循环、自环，以及超过该端口数的图没有移植。MimIR 中每个逻辑顶点编码为带私有身份 tag 的操作调用，每条使用边经过一个表示输出端口的额外调用；输入端口对应参数 Tuple 的位置。该编码防止 hash-consing 合并不同顶点，但使 MimIR 的物理节点数大于另外两库，表中的 `nodes` 均指逻辑顶点数。

## 共同 benchmark

环境：Intel Core i5-14600KF，进程固定到 CPU 2；C++ GCC 15.2.1、`-O3 -DNDEBUG`，Rust release、rustc 1.99.0-nightly，Boost 1.91。每项独立进程运行 7 次，取中位数；无预热。外部仓库未修改。

共 13 个输入配置、8 种引擎/表示组合、728 条样本。每次运行均与独立 Python oracle 比较**完整的有序顶点映射集合**，同时拒绝重复结果；不依赖数量或 hash 判断等价。

- `chain`：若干条长度 12 的独立链，匹配长度 4 的链。
- `shared-dag`：若干份上述四节点共享 DAG。
- `wrong-port`：把共享 DAG 的一条源端口改错，结果应为空。
- `many-chain`：同一目标图中匹配长度 2 至 8 的七个链模式。
- `overlap`：移植测试中的 100 顶点、三个重叠匹配。

以下为最大规模的**搜索时间中位数，单位 ms**：

| 工作负载 | 逻辑顶点 / 结果 | MimIR Indexed | GHL VF2 | GHL VF3 | portmatching naive | portmatching many |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain | 1536 / 1152 | 10.524 | 3.745 | 5.162 | 0.966 | 2.282 |
| shared-dag | 512 / 128 | 2.160 | 0.292 | 0.701 | 0.241 | 0.467 |
| wrong-port | 512 / 0 | 1.427 | 0.238 | 0.465 | 0.197 | 0.324 |
| many-chain | 1536 / 7168 | 104.756 | 30.859 | 40.149 | 7.518 | 14.317 |

七模式链的规模变化：

| 目标顶点 | MimIR Indexed | GHL VF2 | GHL VF3 | portmatching naive | portmatching many |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 6.874 | 0.474 | 2.053 | 0.497 | 0.900 |
| 384 | 25.622 | 2.972 | 8.070 | 1.910 | 3.549 |
| 1536 | 104.756 | 30.859 | 40.149 | 7.518 | 14.317 |

本次七个相关的小模式中，portmatching 的共享自动机没有胜过 naive。此结果不能推广到更大、更复杂的模式库。GHL VF3 也没有胜过 VF2，不能仅凭算法名称推断更快。

计时拆为 `host_ns`（构造目标表示）、`prepare_ns`（构造模式和显式准备）、`search_ns`（枚举并保留库原生结果）。输入文件读取、结果映射解码、输出文件写入和 oracle 校验不计入搜索。MimIR 的 GraphView 建立计入准备；其每次 `find_matches` 内部分析/索引准备、GHL 调用内部准备仍包含在搜索时间里。portmatching 自动机编译计入准备，因此同时保存 `ready_search_ns = prepare_ns + search_ns` 和包含建图的 `total_ns`。

例如最大七模式链，包含准备的耗时分别为 MimIR Indexed 106.116 ms、GHL VF2 30.868 ms、GHL VF3 40.159 ms、portmatching naive 7.697 ms、many 14.607 ms。结果没有因分离显式准备而反转。

这些是当前适配与实现的测量，不能据此单独判定底层算法优劣：表示大小、语言、分配方式、原生结果格式均不同，MimIR 搜索还包含重写读取集合的生成。本次没有测量依赖类型 join、重写发布，也没有测大型多模式库。

## MimIR 原生工作负载

另用已有 [pattern_bench.cpp](../../test/pattern_bench.cpp) 在相同 Release 构建、相同 CPU 上测量，每项 7 次。规模 640，单位 ms：

| 工作负载 | Exhaustive | Indexed | 加速比 |
| --- | ---: | ---: | ---: |
| parallel | 31.687 | 1.357 | 23.4× |
| chain | 3.215 | 3.157 | 1.02× |
| diamond | 55.922 | 1.569 | 35.6× |
| spine | 34.850 | 4.385 | 7.95× |
| fanout | 41.118 | 1.955 | 21.0× |

Indexed 对多根共享输入有明显收益；单根长链的候选数量没有减少，640 节点时仍约 41.7 万 steps。这里比较的是当前实现的两个搜索模式，不是 master 与当前分支的前后对照。

## 已完成的表示实验

为避免把一种降低方式当成引擎能力的上限，另测了 `flat-*` 适配：每个节点结构只描述一次，以 `p.all(vertex_capture, structure)` 为根，边只引用共享 capture。原 `indexed`/`exhaustive` 适配则在节点 capture 内递归嵌套上游结构。两者使用同一个未修改的匹配器，完整映射一致。

最大七模式链的 Indexed 搜索从 104.756 ms 降到 88.400 ms（约 15.6%）；shared-dag 从 2.160 ms 变为 2.697 ms，wrong-port 从 1.427 ms 变为 2.195 ms。Flat + Exhaustive 的七模式链达到 4135.406 ms，明显退化。实验不构成通用优化，也没有接入 Rule lowering；按用户要求不再推进。

嵌套版本的长度 8 多根链会耗尽默认 `max_depth=256`，所以共同 benchmark 显式设为 4096。节点分开描述的对应回归测试使用默认预算。共同 benchmark 的其他预算为每模式 1 亿 steps、1 亿候选、100 万结果；所有搜索均 Complete。

## 数据与复现

- [原始 728 条样本](results/2026-11-02/samples.csv)
- [104 组中位数和工作量计数](results/2026-11-02/medians.csv)
- [原生工作负载](results/2026-11-02/native-medians.csv)
- [源码版本、哈希、环境和预算](results/2026-11-02/metadata.json)

从 MimIR 仓库根目录执行，`<release-build>` 应为已配置的 Release 构建目录，启用 `BUILD_TESTING`：

```sh
cmake --build <release-build> --target mim-portgraph-bench mim-pattern-bench -j4
python3 bench/graph_matching/build_references.py \
  --portmatching /workspaces/compile/portmatching \
  --ghl /workspaces/compile/graph-hook-library \
  --build /tmp/mimir-reference-bench
taskset -c 2 python3 bench/graph_matching/run.py \
  --mimir <release-build>/bin/mim-portgraph-bench \
  --ghl /tmp/mimir-reference-bench/ghl-bench \
  --portmatching /tmp/mimir-reference-bench/port/target/release/mimir-port-bench \
  --output /tmp/mimir-port-results --components 8,32,128 --repeats 7
taskset -c 2 python3 scripts/bench-patterns.py \
  <release-build>/bin/mim-pattern-bench --sizes 80,320,640 --repeats 7
```

首次外部构建需要公开 Rust 依赖；缓存完整时可给构建脚本传 `--offline`。依赖锁文件保存在本目录。构建脚本只在指定临时目录提取 GHL 的 VF3 头文件，不修改 Boost 安装或外部仓库。
