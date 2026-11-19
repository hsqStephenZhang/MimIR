# GPU 协议的线性类型与依赖索引：设计（草案）

状态：草案，2026-11-19，待审。
对应分支 `refactor/tile-engine`（`bab37d8f5` 之后）。

## 1. 目标

两轮外部审阅（`tile-law-design.zh-CN.md` §29，以及 codex 的架构审阅）都指出，GPU 异步协议的“类型安全”目前名不副实。

协议值（槽令牌 `Held`/`Vacant`、工作组超步 `GM`、带可见性的 `SharedBuf`）已经是带索引的类型，最后由 `gpu.erase_phases` 擦除，没有运行时代价。但两条最关键的性质都不由类型系统给出：
- “至多 / 恰好使用一次”由 `gpu.check_sync` 在 PE 之后数图上的使用次数；
- “读的是该槽的当前代”由 `same_slot` 手写的线性算术判定。

本设计把这两条变成：
- 核心里一个**通用的线性性检查**，所有插件共用；
- **依赖类型的定义等价**，由 normalizer 判定。

同时给共享内存区域加 typestate，并给出一个可以机械化证明的小模型。

非目标：
- 不把搜索、融合、use-def 分析搬进 Mim。§10 的实测是：删掉 Mim 里的调度解释器后，加载 `plugin tensor;` 从 12.2 s 降到 1.2 s，`law_wgmma_ws` 从 13.4 s 降到 1.6 s。
- 不在本阶段做收敛性（uniformity）类型（见 §8）。

## 2. 现状

### 2.1 类型

`src/mim/plug/gpu/gpu.mim`：

```
axm GM: [g e: Nat, p: GroupPhase] → *;                         // 工作组 g 第 e 个超步的阶段 p
axm SharedBuf: [g e: Nat, v: Vis, r: Nat, s: «r; Nat», T: *] → *;
axm Held: [depth: Nat] → [Idx depth, Bool] → *;                 // 持有某槽，live 为假是“空令牌”
axm hold:   {depth} → [mem.M 0, s: Idx depth] → [mem.M 0, Held depth (s, tt)];
axm unheld: [depth] → Held depth (core.idx 0 depth 0, ff);
axm retire: {depth} → [mem.M 0, s: Idx depth, live: Bool, Held depth (s, live)] → mem.M 0;
axm Vacant: [depth] → Idx depth → *;
axm vacate / occupy …
axm read_slot: {P, depth, H} → [ring: «depth; P», s t: Idx depth, H (t, tt)] → [P, H (t, tt)];
```

协议（`ring_fold_at`、`ws_fold_at`、`SyncOf`、`tma_ring` 等）是 Mim 里的高阶函数，以 `SyncOf depth H V` 为参数，所以同一 fold 可以接不同的同步实现。

### 2.2 检查

`gpu.check_sync`（`src/mim/plug/gpu/phase/check_sync.cpp`，约 280 行）在 `tensor.lower_to_mem` 之后、`gpu.erase_phases` 之前运行（`src/mim/plug/opt/opt.mim`）。它对 PE 之后的图：
- 收集类型含 `GM`/`SharedBuf`/`Held`/`Vacant` 的 Def；
- 按一套白名单规则判定其使用：
  - 不得进 `Pack`；
  - 部分取用只能用字面量下标，且不得与整体使用同时发生；
  - 选择只能在 mutable 之间进行，各分支的使用互斥；
  - 捕获它的 mutable 必须只运行一次（直接调用、`cps2ds_dep` 调用一次、或分支选择）；
- 另外核对 `read_slot` 的槽号（`same_slot`，模深度的线性形式）。

### 2.3 已知缺口（审阅给出的反例）

| 缺口 | 反例 | 根因 |
|---|---|---|
| nominal 包装 | 把 `Held` 包进 `nom Box (T: *) = T`，两次交还都通过 | `has_slot` 只递归 `Sigma`/`Arr`，线性性不随类型构造传播 |
| 槽号算术 | 深度 3、`t = (3·i + j) % 3`、`s = j`，若 `i` 大到使 `3·i + j` 绕回 2⁶⁴ 则不成立 | `same_slot` 在 C++ 里重做了一遍算术，语义（Nat 是数学自然数还是 u64）没有写明 |
| 环的身份 | 两个同深度的环，令牌类型相同，可以混用 | 索引里没有环 |
| 代（generation） | 读到上一轮填充的面板（mbarrier parity 错位） | 索引里没有代；现在只有 `ring_safe` 对字面量块数做至多 16 步的有界模拟 |
| 轮转的环 | `ws_fold`、逐块轮转的 `ring_fold` 读 `slots#0`，不受 `read_slot` 覆盖 | 位置与槽号的关系没有在类型里出现 |
| 共享区域复用 | 叠放的环槽与 epilogue 暂存区，上一轮 epilogue 还在读，下一轮 prologue 已写（已修，靠人工加 fence） | 区域的生命周期不是类型 |
| 部分 CTA 当全组 | 只覆盖部分线程的 warpgroup 生成全组屏障（编译失败） | 没有收敛性/参与者类型 |

## 3. 设计原则：什么进类型，什么留在 C++

| 放进 Mim / 类型 | 留在 C++ |
|---|---|
| 协议（环、WS、ping-pong），用户会换的部分 | use-def 遍历、融合、别名、逃逸分析 |
| 契约：令牌的索引、线性性、typestate | 搜索、候选枚举、代价模型以外的启发式 |
| law（参考语义、合并许可） | **一个通用的**线性性检查器、通用的 normalizer 规则 |

衡量标准有两条：
- 新增的 C++ 是否与插件无关、规则是否可以一句话陈述；
- 新增的 Mim 是否只是类型和索引，而不是新的解释器。

## 4. T1：核心里的线性性

### 4.1 表示

候选方案：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 线性 universe | `Type` 加一位，`*ₗ` 与 `*` 并存，需要 `*ₗ`/`*` 的子类型或强制规则 | 改 `Type`、`World` 的 universe 计算、所有 `*` 的使用点；对 hash-consing 和推断影响面大 |
| B. 线性类型构造器登记 | 插件在 C++ 里登记“哪些 axiom 构造线性类型”，可带一个判定函数（例如 `SharedBuf` 仅当 `v ≠ Published` 时线性）；复合类型的线性性由结构推出 | 不改核心数据结构；登记在 C++，Mim 源码里看不见 |
| C. Mim 标注 | 在 axiom 声明上加标注，例如 `axm Held: … → *, linear;`，解析器把它记进 annex 元数据 | 要改解析器、`.mim` 元数据和头文件生成；可读性最好 |

**建议 B 起步，C 作为后续的语法糖。**
- B 是登记表加一个通用检查，改动局限在新文件里。
- 线性性由结构推出：
  - `Sigma`、`Arr`（元素线性则线性；`Pack` 一律拒绝）、`Nom`（展开后判定）、`Inj`/`Join` 的成员，只要含线性成分即线性；
  - `Pi`：捕获线性值的 lambda 本身是线性闭包，见 §4.3。
- 这样 nominal 包装的缺口会自动消失。

### 4.2 资格：线性与仿射

- **线性**（恰好一次）：`GM`、未发布的 `SharedBuf`、`Vacant`。
- `Held` 目前允许 fold 结束时留下最后一个槽不交还（由 `settle` 收尾），是**仿射**的。
  - 建议改成线性：`settle` 显式消耗最后的令牌。
  - `unheld` 是一个合法的“空令牌”，可以被 `retire` 消耗，但 `read_slot` 只收 `live = tt`（已实现）。
- 登记时给出资格：`Linear` 或 `Affine`。检查时 `Affine` 允许零次使用。

### 4.3 检查规则

新的核心 phase `core.check_linear`（或 `libmim` 中的 `mim/phase/linear.h`）对 PE 之后的图检查。对每个类型为线性的 Def `x`：

1. **使用计数**：`x` 的使用者集合中，两两“互斥”的组内至多一个。线性资格还要求每条完整控制路径上恰好一个。
   互斥的判定沿用 `check_sync` 已有的规则：
   - 同一个 `Extract` 选择下不同 mutable 分支里的使用；
   - `cps2ds_dep` 只调用一次的 continuation。
2. **不得复制**：
   - 不得出现在 `Pack` 体内；
   - 不得以非字面量下标从含线性成分的元组里取用；
   - 部分取用（字面量下标）与整体使用不得在同一路径上同时发生。
3. **闭包**：捕获 `x` 的 mutable 视为一个使用，而且它自己成为线性的，即只能被调用一次。“只调用一次”沿用 `check_sync` 的 `called_once`：直接调用、作为 `cps2ds_dep` 的参数、或作为互斥分支之一。
4. **作用域**：在 mutable `m` 中产生的 `x`，不得在 `m` 内部一个可能重复执行的 mutable 中被使用（现有规则）。
5. **擦除前提**：线性检查必须在 `erase_phases` 之前、在所有可能复制或丢弃 Def 的变换之后运行。目前的流水线位置（`lower_to_mem` 之后）满足这一点，但要写成不变式：此后、擦除之前，不得再有 PE、内联或 CSE。

### 4.4 与 hash-consing 的关系

MimIR 对不可变值做 hash-consing：两个结构相同的应用是同一个 Def。如果两次 `hold (m, s)` 的 `m` 也相同，它们会合并成一个令牌，从而被当作“用了两次”。现在之所以没出问题，是因为每个产生令牌的 axiom 都带一个新的 `mem` 参数。

本设计把这一点写成登记约束：**每个产生线性值的 axiom 必须以一个 `mem.M` 为参数**，`unheld` 这类常量令牌除外，它们由资格 `Affine` 加“只能被特定 axiom 消耗”覆盖。登记时检查签名，不满足就报错。

### 4.5 `mem.M`

`mem.M` 本质上也是线性的效果令牌。但现有的 `mem.add_mem`、`mem.fresh`、各插件的 `bot` mem 占位都依赖它可以被复制或凭空产生。本阶段**不**把 `mem.M` 登记为线性，只在文档中注明这是将来的统一方向。

### 4.6 迁移

- `check_sync` 的槽令牌部分（白名单、`has_slot`、`is_slot_token`）由 `core.check_linear` 取代。
- `check_sync` 只保留 GPU 特有的规则：工作组构造不嵌套进入、令牌不跨 kernel 边界、已发布面板与 `claim` 的冲突。
- 现有测试 `lit/error/ring_token_*.mim`、`ring_slot_*.mim` 全部保留，再加入审阅给出的反例：
  - nominal 包装后交还两次；
  - `unheld` 读槽；
  - 闭包捕获令牌后调用两次。

### 4.7 估计

- 新增：核心检查约 250 行 C++，登记接口约 50 行。
- 删除：`check_sync` 约 150 行。
- 不改任何 `.mim` 协议代码；唯一的协议改动是 §4.2 让 `settle` 消耗最后的令牌。

## 5. T2：索引里写上环、槽和代

### 5.1 环的身份

把环本身作为令牌类型的索引：

```
axm Held:   {P: *, depth: Nat} → [ring: «depth; P»] → [s: Idx depth, g: Nat, live: Bool] → *;
axm Vacant: {P: *, depth: Nat} → [ring: «depth; P»] → [s: Idx depth, g: Nat] → *;
```

`ring` 是槽面板组成的元组，本身就是一个值。MimIR 是依赖类型的，类型可以依赖它。
- 两个不同环的令牌类型因此不同，混用就是类型错误。
- 擦除时索引一起消失。

代价：
- `SyncOf depth H V` 的 `H`、`V` 要多带环参数；
- `ring_fold_at`/`ws_fold_at` 把 `ring` 传给 `await`/`acquire`。

这些都是 Mim 里的机械改动。

### 5.2 槽号相等变成定义等价

现在 `read_slot` 带两个槽号 `s`、`t`，由 C++ 核对二者相等。改为：

```
axm read_slot: {P, depth, ring} → [s: Idx depth, g: Nat, Held ring (s, g, tt)] → [P, Held ring (s, g, tt)];
```

令牌的类型必须是 `Held ring (s, g, tt)`，其中 `s` 就是要读的槽。
- 在 `ring_fold_at` 的展开环里，`await (m, block)` 给出的令牌类型是 `Held ring (idx (block % depth), …)`，读的位置是字面量 `j`，`block = i · depth + idx2nat j`。
- 类型检查要求 `idx depth (core.nat.rem (i · depth + idx2nat j, depth)) ≡ j`。

为此在 `core` 的 normalizer 里加一条通用规则：

> 对 `core.nat.rem (x, d)`（`d` 为字面量），把 `x` 展开成字面量倍数之和。若除一个 `idx2nat (j)`（`j: Idx d`）以外，所有项的系数和常数都是 `d` 的倍数，则结果是 `idx2nat j`。
> 相应地，`core.idx 0 d (idx2nat j) ≡ j`（`j: Idx d`）已经成立或可加。

这条规则**与 GPU 无关**，可以复用到循环展开后的任何取模下标。随后 `same_slot` 被删除。

**语义前提**：这条规则把 `core.nat` 当作数学自然数。代码生成把 `Nat` 落成 64 位整数，所以只有在不溢出时它才忠实于实现。这正是审阅指出的绕回问题，现在把它变成一条显式的全局假设：
- 写进 `core.nat` 的文档；
- 在 `ring_fold` 处加一个可检查的上界：块数 `count` 是 I64，`i · depth + j < count + depth` 不会溢出。

### 5.3 轮转的环

`ws_fold` 和逐块轮转的 `ring_fold` 每步把槽元组左移一位，然后总读位置 0。有两种处理：

1. **类型化的轮转**：增加 `axm rot: {P, depth} → [n: Nat, ring: «depth; P»] → «depth; P»`，以及 normalizer 规则 `(rot n ring)#k ≡ ring#(idx ((n + k) % depth))`、`rot n (rot m r) ≡ rot (n + m) r`。
   - 携带的状态从“当前轮转后的元组”改成“原元组 + 已转的步数 n”，读位置 0 即读 `ring#(n % depth)`，令牌类型 `Held ring (idx (n % depth), …)` 由 §5.2 的规则匹配。
   - 擦除后 `rot` 落成现在的轮转代码。
2. **不轮转**：一律采用按圈展开（`turns = tt`），轮转仅作为寄存器压力的优化。§28 的实测说明它对 attention 有 16% 的影响，所以轮转不能取消。

建议方案 1。

### 5.4 代

令牌再带上代 `g`：第 `n` 次填充的槽是 `n % depth`，代是 `n / depth`。
- `acquire (m, n)` 给出 `Vacant ring (idx (n % depth), n / depth)`；
- `await (m, n)` 给出 `Held ring (idx (n % depth), n / depth, tt)`；
- `release` 消耗 `Held ring (s, g, live)`，`acquire` 下一次给出的是 `g + 1`。

这样“读到上一代面板”就变成类型不匹配：比如 parity 算错、`exact` 环少填一块。这部分覆盖了 `ring_safe` 的有界模拟。

问题在于：代依赖运行时的块号，类型等价要靠 §5.2 那样的 `div`/`rem` 规则才能判定。`(i · depth + j) / depth ≡ i` 也需要同一条规则的 `div` 版本。

这一项（T2b）放在 5.1–5.3 之后，先评估规则是否足够。

### 5.5 估计

- core normalizer 规则约 80 行。
- `gpu.mim` 协议签名与 fold 的改动约 60 行 Mim。
- `LowerToMem` 构造 `SyncOf` 的调用点约 20 行。
- 删除 `same_slot` 和 `nat_form`。

## 6. T3：共享内存区域的 typestate

`gpu.panel_scope g` 让构造 `g` 的面板与其他构造的面板叠放。§27 的叠放和 §29 的修复说明，叠放是否安全取决于“另一组面板的最后一次读是否已经完成”，而这目前靠人工插入 fence。

设计：

```
axm Region:  [g: Nat, st: RegionState] → *;          // st ∈ {Live, Dead}，线性
axm region_enter: {g} → [mem.M 0, Region (g', Dead) …] → [mem.M 0, Region (g, Live)];
axm region_leave: {g} → [mem.M 0, Region (g, Live)] → [mem.M 0, Region (g, Dead)];   // 内含 barrier
```

- 分配一个与 `g'` 叠放的面板时，必须交出 `Region (g', Dead)`。
- `Dead` 只能由 `region_leave` 产生，而 `region_leave` 包含一道工作组屏障，异步读写的区域还要先等这些操作完成。
- 上一轮 epilogue 的暂存区必须先 leave，下一轮的环才能 enter，§29 的 bug 就成了类型错误。

在 `LowerToMem` 里：
- 每个叠放组的进入和离开要把 `Region` 令牌串起来；
- 跨 tile 循环时，令牌作为循环携带的状态。

工作量比 T1/T2 大，放在其后。

## 7. T4：一个可以机械化证明的小模型

目标：给论文提供“类型安全”的确切含义。

范围：
- 单个 CTA；
- 一个 TMA 生产者（或 cp.async 填充），1–2 个 wgmma 消费者；
- 深度为 `d` 的环，`lag ∈ {0, 1}`。

做法：
- 在 Lean 4 中定义：
  - 操作语义：mbarrier 的 phase/parity、异步拷贝完成、wgmma 的提交与等待；
  - 带线性令牌的协议演算：§4 的使用规则 + §5 的索引。
- 证明两条性质：
  1. 每次读面板时，读到的是该槽当前代的填充；
  2. 对一个槽的覆盖发生在它上一代的最后一次读完成之后。
- 写明信任边界：
  - PTX/ptxas 遵守内存模型（§13 记录过 ptxas 把读累加器的指令调度到 wgmma 等待之前，这一类不在证明范围内）；
  - mbarrier 和 TMA 的硬件语义按 PTX 文档建模。

不在范围内：
- 无死锁（progress）：需要参与者计数与公平性假设，可以作为后续工作；
- 编译 lowering 本身的保义。

## 8. 以后：收敛性

只覆盖部分 CTA 的 warpgroup 被当作全组（§29），本质是缺少“哪些线程到达这里”的类型。可以参考 Descend 的执行资源和 Prism 的集合执行粒度：
- 给值加一个一致性的模态（warp 一致、工作组一致）；
- 屏障只允许出现在工作组一致的上下文中。

这一项影响面大，不在本轮。

## 9. 证据表（对应论文主张）

完成 T1/T2 后，每类错误由哪一层排除：

| 错误类别 | 现在 | T1/T2 之后 |
|---|---|---|
| 令牌用两次 | `check_sync` 白名单 | `core.check_linear`（通用规则） |
| 令牌藏进 nominal、闭包 | 漏检 | 线性性随结构传播 |
| 读错槽（展开的环） | `same_slot`（C++ 算术） | 类型等价 + 通用 normalizer 规则 |
| 读错槽（轮转的环、`ws_fold`） | 不检 | `rot` 的类型规则 |
| 混用两个环 | 不检 | 索引含环 |
| 读到上一代 | `ring_safe` 有界模拟 | 索引含代（T2b） |
| 叠放区域的读写重叠 | 人工 fence | typestate（T3） |
| 屏障在发散分支里 | outline 报错 | 以后的收敛类型 |
| ptxas 重排 | 不在范围 | 不在范围（信任边界） |

## 10. 风险

- **编译时间**：依赖索引会让类型更大，normalizer 规则要在每个取模表达式上运行。需要测加载 `plugin gpu;` 和典型 bench 的编译时间，目标是增加不超过 10%。
- **类型推断**：`ring` 作为类型索引后，`SyncOf` 的部分应用和隐式参数可能推不出来，需要显式标注。
- **normalizer 规则的正确性**：§5.2 的规则依赖“Nat 不溢出”的全局假设，要在文档和 `ring_fold` 的上界检查里同时写明。
- **维护**：新增 C++ 都在核心、与插件无关，GPU 插件的 C++ 反而变少；协议改动都在 Mim 的类型签名上，不引入新的解释器代码。

## 11. 计划

| 步骤 | 内容 | 验收 |
|---|---|---|
| 1 | T1：登记接口 + `core.check_linear`，迁移 `check_sync` 的槽令牌部分，`settle` 消耗最后的令牌 | 现有 lit 全部通过；三个新反例被拒；73 个 bench 设备 IR 不变 |
| 2 | T2a：§5.2 的 normalizer 规则，删除 `same_slot` | 同上；`ring_slot_*` 测试的报错由类型检查给出 |
| 3 | T2a：环索引与 `rot` | 两个环混用、轮转环读错槽被拒 |
| 4 | T2b：代 | parity 错位的反例被类型拒绝；对照 `ring_safe` |
| 5 | T3：区域 typestate | §29 的叠放反例被拒，去掉人工 fence 后仍正确 |
| 6 | T4：Lean 模型 | 两条定理，信任边界说明 |

## 12. 第一轮审阅的发现（codex，未完成）

codex 的审阅在给出报告前被中断。它写好并跑完的探针在 `/tmp/codex-types/`，结果如下。被当前 `check_sync` 接受、本应拒绝的有：

| 探针 | 内容 | 对本设计的影响 |
|---|---|---|
| `nom` | 令牌包进 `nom Box (T: *) = T` 后交还两次 | §4.1 的结构传播覆盖 |
| `join` | 令牌注入 `T ∪ Nat`，`match` 后用两次 | 传播规则必须穿透 `Join`/`Inj`/`Match`，写成明确规则 |
| `same-app` | 同一个 `m` 上两次 `gpu.hold (m, s)`，hash-consing 合并成一个令牌 | **§4.4 的前提不成立**：`mem.M` 本身可以复制，“每个令牌由不同的 mem 产生”无从保证。要么把 `mem.M` 纳入线性检查（§4.5 不能再推迟），要么令牌带上唯一来源 |
| `unit-dictionary` | `read_slot` 的 `H` 实例化为 `λ _ = []`，不持任何令牌读槽 | `read_slot`（和 `SyncOf`）不能对任意 `H` 多态；`H` 必须是登记过的线性构造器 |
| `forged` | 不经 `await`，直接 `gpu.hold` 造令牌读槽 | 线性只管使用次数，不管构造。`hold`/`vacate` 必须是协议私有的能力：只有 `await`/`acquire` 这类状态转换能产生令牌 |
| `drop-vacant` | `Vacant` 被丢弃 | §4.2 把 `Vacant` 定为线性即可 |
| `confluence` | 核心现有规则 `(c·x) % b = 0`（`b | c`）与先代入、再按 u64 折叠的结果不同（`(3·i) % 3` 得 0 或 2） | 核心 normalizer 已经不合流。Nat 的语义（数学自然数还是 u64）必须在核心定下来，§5.2 的规则不能只靠 GPU 侧的约定 |

被正确拒绝的有：`for-body`、`for-exit`（在可能重复执行的循环体或出口里用令牌），`return-box`（返回捕获令牌的闭包后调用），`same-app-diverge`，`permutation`（计算下标取令牌）。

据此修改计划：
- T1 增加两条：
  - 令牌构造只能来自登记的状态转换（能力私有化）；
  - `mem.M` 的线性性与 T1 一起处理，至少保证产生令牌的 mem 不被复制。
- T2 之前，先在核心决定 Nat 的语义，并让 `normalize_nat` 的规则与之一致。
