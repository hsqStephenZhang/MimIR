# portmatching 风格约束执行实验

实验分支：`exp/portmatching-constraints`。开始前的源码补丁、部分文件快照和原始二进制保存在 `/tmp/mimir-portmatching-experiment/baseline`。

本版引入一个独立的单模式后端，借鉴 portmatching 的**预编译约束、绑定依赖和按绑定值查询候选**。它没有移植 Rust 实现，也没有实现 ManyMatcher 的多模式共享自动机。

## 执行机制

[constraint_match.cpp](../../src/mim/pattern/constraint_match.cpp) 将每个根降低为顺序指令：

| 指令 | 含义 |
| --- | --- |
| Check | 检查 kind/arity/flags、Exact 或 typed Literal |
| Operand | 沿指定 operand 边加载值 |
| Type | 沿类型边加载值 |
| Capture | 绑定紧凑 capture ID，或检查已有绑定身份 |

同一个 capture 内的结构在单根指令中只展开一次；后续出现保留身份检查。执行指令时使用复用的 slot 数组，没有递归 operand visitor 或 `std::function` continuation。

一次搜索分为三步：

1. 对每个根的候选执行顺序指令，将成功候选及其 capture 写入平坦的行表。
2. 为多个根共有的 capture 建立 `绑定值 → 候选行` 索引，选择一次根连接顺序。
3. 按已绑定列查询最小候选桶，检查其余列和根的单射约束；新增绑定用 trail 撤销。过程中不生成祖先集合，不为多集合求交分配临时容器。

这与 portmatching 的固定端口路径查询存在区别：MimIR 先物化每个根的局部匹配关系，再做索引 join。这样也能处理类型路径上、位于 GraphView 原始节点集合以外的参数。行表和索引每次搜索重建，尚未实现增量更新。

例如 A 的类型 `Tensor([m,k],T)`、B 的类型 `Tensor([k,n],T)`：A/B 行表均包含 `k,T` 列，绑定 A 后可据此查 B；不需要先枚举全部 A/B 组合再检查类型。测试中 24 对独立根，旧 Indexed 尝试 **600** 个根候选，新后端尝试 **48** 个，完整映射和重写读取集合一致。这里候选计数指 join 阶段，线性的候选表构造另计为 steps。

## 接口及接入范围

```cpp
auto found = find_matches(pattern, graph, {.mode = GraphSearchMode::Compiled});
```

也可显式复用编译计划，将编译放到搜索计时以外：

```cpp
CompiledGraphPattern plan(pattern);
auto found = plan.find(graph);
// found.used_compiled 标记本次是否实际使用编译后端。
```

支持 Any、Exact、Literal、Node、Capture、All、Type、Where 和语义 use 关系。Where 在所有根绑定完成后检查，仍保守记录整份 snapshot 的读取依赖。

Either、Not、Range、过大的指令展开，以及运行时需要结构检查的 opaque 节点，回退到原 Indexed 解释器。运行时回退扣除已消费的步数预算，不返回部分编译结果。`is_compiled()` 只说明静态计划可编译，实际回退应看 `used_compiled`。benchmark 拒绝将回退结果标成 compiled。

类型检查提前后，部分旧解释器的 Unknown 可以被证明为 Complete/无匹配：例如 Lam 的类型已不符合要求，新后端不必再进入 opaque Lam 的结构。新增回归测试记录此精度差异；必须检查 opaque 结构才能决定的情况仍回退。

PatternProgram append、根/关系修改、alias 策略变化和 World 重建会使计划失效。计划不拥有 PatternProgram、GraphPattern 或源 World。

本实验分支让 `graph_rule_app` / `graph_rule_app1` 使用新后端并保留回退。低层 API 的默认模式仍是 Indexed；单根 `rule_app` 和 Tensor 的直接匹配客户端未切换。Rule 语法、原始 Tuple/Pack 匹配约定、重写验证及 World 发布机制未改变。

结果仍包含完整 captures、matched nodes 和读取集合；没有关闭重写安全元数据。枚举顺序可与旧后端不同，规则列表的优先次序保持不变。

## 正确性验证

- 全部单测：**223/223，6949 个断言**。
- Tensor 测试：**16/16，143 个断言**。
- Rule lit：**11/11**，包含类型 join、nominal Tensor、guard、输出回读。
- 新增 13 组后端测试：32 个固定种子的随机小 DAG、完整映射/captures/读取集合差分、类型边 join、use 关系、alias、Unknown 精度、预算、静态回退、计划及 World 生命周期。
- 新后端与差分测试 TU 用 ASan/UBSan 编译：**13/13，940 个断言**；既有 libmim 没有整体重编为 sanitizer 版本。环境不能完成 LSan，运行时设置 `detect_leaks=0`。
- TVM DPL、PDLL、P5 和端口图测试的原有多模式循环加入 Compiled，对 Unsupported 情况同时覆盖解释器回退。

`typejoin.mim` 的输出回读原先使用 `%t.mim`，文件名超过 MimIR 的 annex/plugin 命名范围。已将输出置于临时目录下的 `typejoin.mim`，保留回读验证。

## 与真实修改前基线比较

2026-11-02，Release，GCC 15.2.1，Intel Core i5-14600KF，CPU 2，每项 7 次取中位数。开始实验前复制原始 runner 和 `libmim.so`，通过独立 `LD_LIBRARY_PATH` 运行；**基线不是新代码中的功能开关**。另测实验分支的 Indexed 模式作控制，其结果与冻结基线接近。

共同端口 corpus 共 13 个配置、7 个引擎、637 条样本。每次与独立 oracle 核对完整有序顶点映射集合并检查重复；所有搜索 Complete，新后端在这些样本中均未回退。以下为最大规模的**搜索中位数，单位 ms**：

| 用例 | 修改前 Indexed | 实验分支 Indexed | Compiled | 相对修改前加速 | portmatching naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| chain，1536 顶点 | 12.547 | 12.532 | 1.912 | 6.56× | 1.173 |
| shared-dag，512 顶点 | 2.597 | 2.591 | 0.242 | 10.72× | 0.289 |
| wrong-port，512 顶点 | 1.700 | 1.699 | 0.068 | 25.01× | 0.240 |
| many-chain，1536 顶点、7 模式 | 124.538 | 124.460 | 15.330 | 8.12× | 9.320 |

不能只看搜索：MimIR 还要构造较大的 IR 表示和 GraphView。计入目标表示构造和显式准备后：

| 用例 | 修改前总计 ms | Compiled 总计 ms | portmatching naive 总计 ms |
| --- | ---: | ---: | ---: |
| chain | 14.796 | 4.097 | 1.277 |
| shared-dag | 3.421 | 1.053 | 0.341 |
| wrong-port | 2.532 | 0.889 | 0.291 |
| many-chain | 127.042 | 17.801 | 9.624 |

Compiled 的显式指令编译计入准备，行表构造和索引建立仍计入搜索；portmatching 的 matcher 编译也计入准备。`ready_search_ns` 同时保留准备加搜索的数据。表中各列独立取中位数，因此不强求逐列相加恰好等于总计。

目前仅共享 DAG 和失败例的搜索时间低于 portmatching naive；计入准备后仍较慢。两者的物理表示、结果格式与安全元数据不同，此次不能据此判定普遍优劣。

## 原生工作负载

已有 pattern benchmark，规模 640，同样重复 7 次，单位 ms。Compiled 通过 `find_matches` 使用，指令编译也包含在搜索内：

| 用例 | 修改前 Indexed | Compiled | 加速 |
| --- | ---: | ---: | ---: |
| parallel | 1.657 | 0.334 | 4.96× |
| chain | 3.809 | 3.273 | 1.16× |
| diamond | 1.893 | 0.371 | 5.10× |
| spine | 5.259 | 3.643 | 1.44× |
| fanout | 2.373 | 0.433 | 5.48× |

单根长链仍受逐结果读取集合的平方遍历影响；新执行计划没有解决这个独立问题。OR/NOT/Range 的编译、多模式自动机、增量维护，以及适配更大真实规则库均未实现。

## 数据与复现

数据目录：[2026-11-02-compiled](results/2026-11-02-compiled/)。其中包括全部样本、中位数、原生用例前后结果、版本/源码/二进制 hash 和测试记录。

当前工作区的实际运行命令：

```sh
taskset -c 2 python3 bench/graph_matching/run.py \
  --mimir /tmp/mimir-reference-bench/mim-release/bin/mim-portgraph-bench \
  --baseline /tmp/mimir-portmatching-experiment/baseline/run \
  --mimir-modes indexed,compiled \
  --ghl /tmp/mimir-reference-bench/ghl-bench \
  --portmatching /tmp/mimir-reference-bench/port/target/release/mimir-port-bench \
  --output /tmp/mimir-portmatching-experiment/results \
  --components 8,32,128 --repeats 7
```

换环境复现时需要在应用实验改动之前独立构建/保存原始 runner 与库，并使用各自的库运行。源码及二进制 hash 见 [metadata.json](results/2026-11-02-compiled/metadata.json)。
