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

## 13. 与 Async Graphene（CC'26）的对照

Hagedorn 与 Grover，"It's about Time: Temporal Abstractions for Asynchronous GPU Tensor Computations"，CC'26，doi:10.1145/3771775.3786277。下文“§x”均指该论文的章节。

### 13.1 它做了什么

- **类型。** 张量类型在 Graphene 的布局 `[shape:stride].dtype|mem` 上叠加四种时间修饰（§4.1）：
  - future `!T`：按完成机制分为 `!group`（commit/wait 组）和 `!mbar`（mbarrier），省略时由产生它的指令沿 def-use 推断；
  - aref `%T`：生产者/消费者共享的缓冲，状态为 full 或 empty；
  - tuple `<T1, T2>`：多个 future 共用一个 barrier，GEMM 上约快 5–10%；
  - iterator `#T`：环形访问，`<<`/`>>` 取偏移。

  GEMM 的主类型是 `#[7].%!<[128,64].fp16|smem, [64,128].fp16|smem>`。附录 Fig. 1 只给出类型的**构成**规则（`Future`、`Aref`、`Tuple`、`Iter` 等），没有使用规则，也没有可靠性定理。
- **到达与参与者。** 线程也是张量：用切片和 `elect_leader` 圈出执行范围，参与谓词由算法 1 从线程张量的 AST 推出，aref 的 barrier 到达数也由此推出（§4.2.1）。
- **流水。** 任务带 `delay`，由算法 2 生成 prologue、稳态和 epilogue（§4.2.2）。
- **硬件。** `async.tma`/`async.mma` 从类型推出 TMA 描述符、坐标、wgmma 描述符，并检查秩、形状、内存空间和目标是否为 future（§4.3）。
- **评测。** H100 与 B200 上 GEMM 达到 cuBLAS 的 98%；FlashAttention（head dim 64/128/256）达到 FA3 的 98%。

### 13.2 它的保证到哪里为止

- **future 只区分“已等待/未等待”。** `!T` 必须 `wait` 才能当 `T` 用，可以防止过早使用。但 future 不是线性的，没有规则阻止重复等待或丢弃。
- **aref 的 full/empty 是运行时状态**（mbarrier 的 phase），不是类型状态。`put`/`get` 的作用域结束时会自动发信号。
- **延迟释放完全手写**（§4.1.2）：`delay_signal = True` 关掉自动信号，再手写 `wgmma.wait_group(delay)`、`iter << delay`、`signal_consumed`。这一步回退的是不是正确的槽、对应的 wgmma 是否已经退役，都不受检查。
- **“by construction”只是一句话。** 论文称“prevents synchronization bugs by construction”（§6），但没有使用规则和定理来支撑。

### 13.3 我们借鉴什么

1. **数据与能力合一。** 面板本身带时间修饰，例如 `Future (ring, s, g) P`；只有 `await` 能把它变成 `P`，不再有游离的令牌。这样 §12 的 `forged`（凭空构造令牌）和 `unit-dictionary`（单位类型冒充令牌）在用户层面不再可能出现。
2. **作用域化的获取与释放。** 用户接口采用 `with_slot ring n (λ panel. body)`，线性令牌只留在协议库内部，§4 的线性检查只需覆盖库代码。这也绕开了 `mem.M` 可复制导致令牌被 hash-consing 合并的问题（§12 的 `same-app`）。
3. **完成机制进类型。** 区分 `group` 与 `mbar`，对应我们的 `async_ring` 与 `mbar_ring`。wgmma 的在途组数（`InFlight lag`）可以作为 `group` future 的计数来检查。
4. **布局进类型。** 把 `layout_expr` 放进 `SharedBuf` 的类型。它的 `async.tma` 只检查秩、形状和空间；我们可以由 F₂ 判定 wgmma 描述符和 TMA 落点与类型里的布局一致。这比它的检查更强。
5. **参与者是线程张量。** 用它的做法，从线程范围推出参与谓词和 barrier 到达数，可以直接解决 §8 的问题，即 §29 中“部分 CTA 被当作全组”。
6. **future tuple 共用 barrier。** 我们的环槽本来就是面板元组，可以作为类型层面的显式标记。

### 13.4 我们可以多做、因而构成差异的

| 方面 | Async Graphene | 本设计 |
|---|---|---|
| 未等待就用 | 类型排除（`!T` vs `T`） | 同（future 修饰） |
| 重复使用、遗漏释放 | 不检查 | 线性令牌，限于库内部（§4） |
| full/empty | 运行时 phase | typestate：`Vacant` → `Held` → 释放（§4、§5） |
| 读到正确的槽和代 | 迭代器维护，不检查 | 环、槽、代作为依赖索引，由类型等价判定（§5） |
| 延迟释放（`delay`） | 手写 `iter << delay` + `signal_consumed` | 由 `InFlight` 见证和索引检查“回退的就是已退役的那一槽”（§5） |
| 叠放区域复用 | 无 | 区域 typestate（§6） |
| 可靠性 | 没有定理 | Lean 小模型中的两条定理（§7） |
| 运行时代价 | 无（编译期抽象） | 无（检查后擦除） |

论文主张据此应写成：在 Async Graphene 的时间修饰之上，加上**线性性、依赖索引和 typestate**，把它留给程序员的那几处检查（延迟释放、槽与代、区域复用）变成可判定的类型检查，并给出模型中的可靠性定理。

性能上它是强基线（H100 上 FA3 的 98%），我们的 attention 只有 cuDNN 的 0.73–0.97，所以论文应以类型为主线，性能只作“不付代价”的证据。

## 14. codex 审阅结论与修订（2026-11-19）

报告全文：`/tmp/claude-7076018/-home-chzh00001-workspaces-MimIR/acdcdd87-d117-42ff-af4a-7559329fd915/scratchpad/codex/types_review.md`，探针在 `/tmp/codex-types/`。结论：方向保留（协议写在 Mim 里、生成 GPU 代码前检查），§4–§7 的安全契约要重写。

### 14.1 四个阻塞项（P0）

1. **核心转换检查把不同槽号当成相等。**
   - `cast_slot (s) (t) (h: Held 3 (s, tt)): Held 3 (t, tt) = h` 通过类型检查，再用两个字面量实例化，就得到 `swap: Held 3 (0₃, tt) → Held 3 (1₃, tt)`（`/tmp/codex-types/closed-slot-review.mim`，已复现）。
   - 原因：`src/mim/check.cpp:342`，`Check` 模式下两个自由的 `Var` 一律视为相等。
   - 影响：§5 的“槽号相等变成定义等价”不成立；删掉 `same_slot` 会撤掉唯一一层独立检查。
   - 修订：资源索引要刚性相等，即只在显式 binder 对应时才相等。先在协议检查器里做带 binder 环境的刚性比较，再评估改核心 `coerce` 对推断的影响。
2. **PE 之后的图上，使用次数不等于动态执行次数。**
   - 部分应用 `sink h` 被共享，作为 callee 调用两次，通过检查（漏报）。
   - 闭包装进元组后只调用一次，却被拒（误报）；`for-exit`（只在正常出口消费）和 `permutation` 也是误报。
   - `same-app` 两次 `hold` 加两次相同 `retire` 全部被合并，“计数”已经无从谈起。
   - 修订：
     - 检查分两次：协议 lambda 的绑定结构建成时，检查一次性调用、资源移交和分支汇合；`LowerToMem` 生成实际的环之后、擦除之前，检查生成代码属于已验证的协议片段。
     - 规则采用 `Γ ; Δ ⊢ e : A`。“参数在体内用一次”和“函数值只能调用一次”是两个量。
     - 近期做法是登记构造器（§4.1 方案 B）加 `For`/`cps2ds`/`fresh`/协议字典的调用摘要；长期语义是 QTT 式的 `Pi` 域重数。
3. **能力来源没有把关。**
   - 隐式参数 `{h: Held …}` 只留下一个 `Hole`，`check_sync` 跳过它，`erase_phases` 照样擦成 `r#s`；直接传 `⊥: Held …` 也被接受。
   - `hold`/`vacate`/`in_flight` 是公开构造器，`read_slot` 返回的面板还能逃出持有令牌的范围。
   - 修订：
     - 引入独立的线性协议前态 `RingState ρ`，令牌只能由消费唯一前态的转换产生；`mem.M` 继续只负责效果排序，不改成线性。
     - 公开构造器全部密封。
     - 面板改用带生命周期的视图（rank-2 风格的 `withPanel`）。
     - 签发验证结果时，能力来源、资格和索引必须全部确定，不允许残留 `Hole` 或 `⊥`。
4. **Nat 的语义两边都不一致。**
   - 核心的 `(c·x) % b = 0` 与先代入、再按 u64 折叠的结果不同；`(c·x)/b` 和 `(x+y)-y` 也依赖不溢出。
   - LLVM 后端给加法、乘法带了 `nuw nsw`：`2⁶³−1 + 1` 不违反 `nuw`，却违反 `nsw`，结果是 poison。
   - 修订：
     - §5.2 的规则不进全局定义等价，只作为协议片段内的算术引理：固定的正深度、`Idx d`、常数系数的仿射项，判定不了就返回 unknown。
     - Nat 的语义要在核心层面单独决定。

### 14.2 其余要点（P1）

- **环的身份。** 把面板元组当作类型索引有两个方向的错误：两个独立的 barrier 协议可以共用同一组面板；同一个环经轮转、封装、循环传参后又会变成不同的项。
  - 修订：`withRing` 引入一个不能逃逸、生成式的身份 `ρ`。`ρ` 可以擦除；用来定位面板的 handle 和游标要保留。
- **代。** `await` 可以自称返回 `Held (ρ, n % d, n / d)`，类型却与真实 parity 无关。
  - 修订：期望的代要从“本次消费的逻辑块”推出，`Held` 只能由配对的 fill 和正确 phase 的完成证据产生。
  - `ring_safe` 的有界模拟还检查“能继续执行”，不能删。
  - persistent 的 `nth` 按每个槽实际填充次数编号，要用另一套不变式。
- **区域 typestate。** `Dead` 必须说明谁对哪块存储的访问已经完成：物理字节区间、CTA/cluster 实例、epoch、读写权限、未兑现的完成义务。
  - 异步发射把借用转给在途操作，匹配的完成事件才归还。
  - WGMMA 要按 group 关联读过的槽：`wait_group 1` 之后只能释放已经不在剩余组里的槽。
  - WS、multicast 和 persistent 各有单独的义务。
- **Lean 模型。**
  - 首个目标：单 CTA、TMA、一个生产者、一个 WGMMA 消费者、任意正深度和任意有限块数；先做 `lag 0`，再做 `lag 1`。
  - 三类事件分开建模：barrier 的代、写完成与可见、读完成与可重用。
  - 定理要量化所有合法交错，并给出能正常 drain 的执行，避免定理空真。
  - **需要对照**：Koska 等 2026-11 的预印本已经在 Lean 中形式化 PTX 子集，并验证了含 TMA、mbarrier、WGMMA 的 Hopper GEMM（arXiv:2611.04447）。只证明一个 Hopper kernel 已经不新；差异在于有类型的协议族、任意深度和块数，以及与高阶生成器的自动连接。
- **替代方案。** 最适合先做的是一个线性的 `RingState ρ`：fold 状态携带生产者和消费者的游标、槽阶段和未完成组。WS 在分叉时拆出各角色的权限，在 drain 处汇合。按 `(ρ, slot, generation, role)` 索引的权限适合接 T3。
- **论文价值。**
  - T1 的登记表、T2 的取模化简与 `rot`、T3 的两状态接口，单独看都偏工程。
  - 较强的主张是：在会做 PE、共享不可变节点的高阶 IR 中，协议抽象特化后仍有可检查的资源解释；由同一个协议不变式，推出任意允许深度下“读到本代”和“覆盖安全”。

### 14.3 修订后的顺序

1. 核心判等与能力的反例先修：刚性索引比较，签发验证结果时不允许残留 `Hole`/`⊥`，密封公开构造器。
   首个验收：
   - 合法的单次闭包调用通过；
   - `swap: Held(0) → Held(1)`、未解的隐式能力读槽、同一个部分应用释放两次，全部被拒；
   - 一个真实的 `LowerToMem` 单 CTA 环仍能通过并擦除。
2. 单 CTA 的操作语义与密封的 `RingState ρ`。
3. 验证一个实际的残余 fold 属于已证协议。
4. 第二个消费者与轮转。
5. 叠放。

§4.7、§5.5 的行数估计作废，改按上面的里程碑估算。每一步报告：误拒、unknown、PE 节点数、峰值内存、编译耗时、寄存器变化。

## 15. 定位：我们是编译器，不是 kernel DSL

MimIR 是编译器：用户写的是 PyTorch 模型（经 `mimir-frontend` 导入），kernel 由编译器生成。前面几节借用了 DSL 论文“留给程序员的事”的说法，需要按这个定位改写。

- **类型检查的对象是编译器自己的输出。** 协议的线性性、槽和代的索引、区域状态，检查的是 `LowerToMem` 生成的实例。检查失败说明编译器有 bug，而不是用户写错了。所以这套类型系统的作用相当于**验证编译器输出**（certifying compilation / translation validation），而不是给用户的安全网。
  - 仓库里已有这类 bug 的实例，都是编译器在叠放或主序上的错误：§29 的跨轮叠放、同一面板两种主序。它们比合成的“用户误用”例子更适合当论文的动机和评测。
  - 诊断面向编译器开发者：报出是哪个 phase、哪个协议实例、违反了哪条义务。
- **对照的轴要分开。**
  - **编译器轴**：Triton 编译器（含自动 WS）、TileLang 的自动部分、IREE、TVM、Tawa（从 tile 程序自动推出 aref）。比较的是能否检查**自动生成的**同步是否正确，以及检查的代价。
  - **kernel 语言轴**：Async Graphene、ThunderKittens、CuTe DSL、Gluon。在这条轴上，“程序员写错”对应的是我们的第二种用法，见下一条。
- **能写 kernel 同样重要。** Mim 本身可以直接写 kernel：`gpu.mim` 里的协议库、`bench/typed_tiles`、`lit/gpu/warpgroup_owned.mim` 这类手写张量级 IR。专家可以写新协议或特殊 kernel，编译器的 lowering 也是用同一套协议写成的。这条路径上，§13 的借鉴照样成立：
  - 作用域化的 `with_slot`；
  - 面板带 future 修饰；
  - 公开构造器全部密封。

  这样，编译器生成的 kernel 和人写的 kernel 走同一套检查。
- **论文主张据此改写为**：同一套类型化的协议既是编译器 lowering 的目标，也是手写 kernel 的语言。类型检查在擦除前验证编译器为每个 PyTorch 模型生成的协议实例。评测用真实的编译器 bug（历史 bug 与变异）说明检查的价值，再用手写 kernel 说明表达能力。
- **对 §14 的影响**：
  - 能力来源的那些漏洞仍要修：`Hole`、`⊥`、公开构造器、面板逃逸。编译器自己的 bug 也会造成同样的效果，比如生成代码时漏传或错传令牌。
  - 但“用户恶意构造”的反例优先级下降；优先级最高的是编译器可能真正犯的错：槽号算错、代算错、叠放区域提前复用、漏掉参与者。

## 16. 第 1 步的实现（2026-11-19，`feat/protocol-types`）

工作区 `~/workspaces/MimIR-types`，分支 `feat/protocol-types`，从 `03ea5ef00` 开出。改动都在 `gpu.check_sync` 和 `gpu.mim` 里：
- **刚性比较。**
  - 凡是参数或返回值的类型含槽令牌的地方，在 PE 之后的图上用 `Check::test_alpha`（Test 模式，不把两个不同的自由变量视为相等）重新比较。依赖元组先用实际的值实例化，再逐分量比较。
  - 涉及两处：调用（`App` 的实参对被调函数的定义域）和函数体（`Lam` 的函数体对其返回类型）。
  - 效果：`swap: Held 3 (0₃) → Held 3 (1₃)` 被拒（`lit/error/ring_slot_cast.mim`）。核心 `coerce` 没有改。
- **来源。**
  - 未填的 `Hole` 和 `⊥` 当令牌用，都被拒（`ring_token_hole.mim`、`ring_token_bottom.mim`）。
  - `gpu.hold`、`vacate`、`unheld` 改为 `priv`，只有 `gpu.mim` 里的环同步能构造令牌（`ring_token_forge.mim`）。
- **闭包。**
  - 部分应用若接收了令牌，它就是持有令牌的闭包。闭包的调用点沿元组、字面量下标投影、函数返回值、只调用一次的函数的参数去追，追到的调用必须两两互斥；追丢了就按“可能多次调用”拒绝（`ring_token_closure_twice.mim`）。
  - `repeats` 先用同一追踪：所有调用点两两互斥、且都不在该闭包内部时，就认为它不会重复运行。原先误拒的“装进元组、取出、调用一次”和“传给只调用一次的函数”现在都能通过（`lit/gpu/ring_token_closure_once.mim`）；传给调用两次的函数照样被拒（`ring_token_closure_apply.mim`）。
- **仍未处理。**
  - `permutation`：计算下标拆开令牌元组，仍保守地拒绝。
  - `for-exit`：只在正常出口消费的令牌。原探针用的是 `gpu.hold`，需要改写后再判定。
  - nominal、`Join`/`Inj` 包装的传播。
  - `drop-vacant`：丢弃 `Vacant`，即线性资格。

## 17. 第 2 步之前要定的事

1. **循环状态不能依赖迭代号。**
   `affine.For` 的状态类型是固定的，所以 `RingState (fills, awaits)` 这种按计数索引的状态，在循环里只能以存在类型隐藏下标，类型就无法把 `await n` 与计数器联系起来。三种取舍：
   - （a）`RingState ρ` 只作为不可伪造的线性能力，管来源，不管计数；
   - （b）给 fold 一个依赖类型的循环组合子（状态类型是 `Nat → *`），由 PE 展开前的类型检查负责；
   - （c）计数由 §14 的残余 fold 验证器检查，即检查生成代码是不是已证协议的实例。

   建议 （a）+（c）。
2. **Lean 工具链。** 本机没有 `lean`/`lake`/`elan`。T4 需要安装 elan，并在用户目录下载工具链。
3. **codex 的代劳。** 让 codex 在工作区里无沙箱改代码被本地权限检查拒绝了，这个容器里它的沙箱又起不来。要把实现任务交给 codex，需要为这条命令加一条权限规则。

## 18. 第 2 步：环的身份（2026-11-19，`b8322375e`）

按 §17 的决定，`RingState ρ` 采用“能力 + 残余验证”。第一部分是环的身份：
- 令牌类型带上环编号：`Held (rid, depth) (s, live)`、`Vacant (rid, depth) s`；`hold`/`retire`/`vacate`/`occupy`/`unheld`、`RingSync`、各环构造器、`ring_fold`/`ws_fold` 都多一个 `rid`。
- `rid` 由 `LowerToMem` 给每个环分配一个新的字面量（`rings_`），在环的整个生命周期里固定，不随迭代变化，所以与 `affine.For` 的固定状态类型相容。
- 两个环的令牌混用是类型错误，前端在构造调用时就拒绝（`lit/error/ring_token_other_ring.mim`）。
- `rid` 只出现在类型里，擦除后消失：73 个 bench 的设备 IR 与改动前完全相同，全量 lit 1053 通过。

还没做的：把 `open` 产生的线性 `RingState rid` 串进 fold 状态，使令牌只能由消费唯一前态的转换产生；以及第 3 步的残余 fold 验证器，它等 Lean 模型确定协议的形状之后再做。

## 19. 第 3 步：具体残余 fold 的验证

单 CTA、单消费者的 TMA 实例已接到 Lean 的可执行 `Ring.fold`。56 个特化实例中，rolled 与 unrolled 逐对一致。28 个 `lag = 0` 实例都比模型少一次退出时的 `wait_group 0`，其余 28 个逐条相等。`LowerToMem` 生成的环可用 `-X tensor:ring-coverage=1` 分类，区域叠放由 NVPTX 的最终布局判定。事件对应、benchmark 计数和剩余证明义务见[验证报告](../proofs/ring/README.md#与残余-fold-对接)。

## 19. 第 3 步结果与 agy 审阅（2026-11-19）

### 19.1 第 3 步（codex，`9ef7aba65`、`f8385cf39`）

- **翻译验证。** 用真实的 `gpu.ring_fold`、`tma_ring`、`tma_fill` 和 `wgmma_batch`，在令牌擦除前记录同步事件，再在主机上执行特化后的循环，与 Lean 的 `Ring.fold` 逐事件比较（`lit/gpu/ring_fold_trace.mim`）。
  - 覆盖 `d ∈ {2,3,4}`、`count ∈ {0,1,d−1,d,2d+1}`、`lag ∈ {0,1}`、展开和轮转两种循环，共 56 个实例。
  - 展开与轮转逐对一致；`lag = 1` 的实例与 Lean 逐事件相等；`lag = 0` 的实例都恰好少了模型在退出时追加的 `wait_group 0`，这是已知差异：非空循环已在最后一次提交后等待，空循环从未提交过组。
- **覆盖统计。** 73 个 bench 里有 11 个环，**满足定理全部前提的为 0**（`bench/tools/ring_coverage.py`）：
  - `cp.async` 5 个；
  - 两个消费 warpgroup 的 TMA 2 个；
  - cluster 2 个；
  - `ws_fold` 1 个；
  - persistent 1 个。

  把普通 TMA GEMM 缩成单个消费 warpgroup 后，F32 输出的环满足前提；F16 输出的环因为 epilogue 与环槽叠放，仍被排除。

### 19.2 agy（Gemini 4 Argon）审阅

报告：`/tmp/claude-7076018/-home-chzh00001-workspaces-MimIR/acdcdd87-d117-42ff-af4a-7559329fd915/scratchpad/agy/review.md`，探针在 `/tmp/agy-review/`。下面的探针我都复现了。

**仍被接受、本应拒绝的：**
- F1：`has_slot` 不进 `Pi`，返回函数的转换绕过刚性比较，可以把 `idle`（`ff`）转成任意环、任意槽的活跃令牌（`pi_cast_probe`）。
- F2：`calls_of`/`repeats` 只检查静态调用点两两互斥，不检查调用点所在的函数或循环是否会重复运行（`loop_curried_probe`）。
- F3：三参数以上的柯里化在第一次部分应用处就停止追踪（`curried3_probe`）。
- F4：`{rid depth}` 是隐式参数，漏写时 `rid` 成为未解的洞，两个环的洞会被合一（`implicit_rid_probe`）。
- F5：面板元组 `«depth; P»` 不带 `rid`，可以用环 0 的令牌读环 1 的面板（`cross_ring_panel_read`）；同一槽连续两次 `await` 会得到两个活跃令牌（`double_await_same_slot`）。
- F6：函数类型的洞或 `⊥`、类型索引里的洞、公开的 `gpu.in_flight`，以及 `GM`/`SharedBuf` 的洞和 `⊥`，都没有被拒。

**被误拒的（F7）：**
- `«2; Last»` 这种含依赖元组的数组（`arr_dep_sigma_fp2`）；
- 经 `cps.cps2ds` 调用一次的持有令牌的 CPS 函数；
- 元组里有两个函数、各调用一次；
- `affine.For` 出口消费外层令牌（`for_exit_probe`）。

**Lean 模型（F9）：**
- `Cell.start` 没有让旧的 `visible` 失效；
- `fill`/`mma` 在语义内部硬编码 `i % d`，表达不了“等了块 `i`，却读了错槽”；
- 单 CTA 非 WS 环里起作用的 `bar.sync`、`wgmma.fence` 和 proxy fence 没有建模；
- `count < d − 1` 时序言会越界填充。

**架构判断：** 在“检查生成代码”的定位下，“PE 后检查 + 私有构造器 + 环身份 + Lean 协议证书”是对的总体结构。但要补两件事：
- 面板句柄要与 `rid` 绑定；
- 刚性比较要穿透 `Pi`、`Arr`、`Nom`，并检查类型索引里没有残留的洞。

此外，残余验证应改用 Lean 已有的 `Ring.Code` 证书逐步检查，而不是要求与 `Ring.fold` 字面相等。

### 19.3 下一轮

1. **检查器（我）**：
   - 刚性比较覆盖 `Pi`、按字面量展开 `Arr`；
   - 类型索引里的洞和 `⊥` 一律拒绝；`gpu.in_flight` 改成 `priv`；
   - `calls_of` 继续追踪柯里化，对每个调用点检查它所在的作用域是否会重复运行，识别 `cps2ds_dep` 和 `affine.For` 的出口；
   - `rid` 改成显式参数；面板元组与 `rid` 绑定。
2. **Lean 模型（codex）**：
   - 修 F9；
   - 扩到 `cp.async`（`async_ring`）和两个消费者，让 bench 里的主力环进入覆盖。
3. **残余验证**：用 `Ring.Code` 证书逐步检查真实残余 fold。

## 20. 第二轮：检查器修复与 Lean 扩展（2026-11-19）

### 20.1 检查器（`3a36603f3`）

按 §19.2 修复：
- 刚性比较穿透函数类型（`mentions_slot`），按字面量下标展开 `Arr`（F1、F7①）。
- 令牌类型里残留未解索引一律拒绝（F4）；函数类型的洞和 `⊥` 一律拒绝（F6）。
- 闭包追踪：
  - 柯里化时，部分应用继续往下追（F3）；
  - 识别 `cps2ds_dep`（F7②）；
  - 元组按位置区分，位置信息随返回值和参数传递（F7③）；
  - `affine.For` 的出口只运行一次（F7④）；
  - 每个调用点要求在闭包产生处与调用点之间没有会重复运行的 mutable（F2）。
- 修掉一处未定义行为：读同一张哈希表的迭代器时又往表里插入，扩容后迭代器失效，导致检查结果时对时错。

新增 lit 测试 9 个；73 个 bench 的设备 IR 不变。仍未处理：
- 面板元组不带 `rid`（F5 的跨环读面板）；
- 同一槽连续两次 `await`（需要 `RingState`）；
- `gpu.in_flight` 仍公开（两个测试直接使用它）。

### 20.2 Lean（codex，`db9aabb61` … `dbe26e329`）

- **单消费者 TMA。**
  - `Cell.start` 清掉旧数据；物理槽改为显式操作数。
  - `live_read_race_free`：在任意可达状态，每个待提交或在途的读 `r` 都有 `visible = some r ∧ tx = none`。
  - `wrong_panel_rejected`：等了正确的块、却读了错槽的程序拿不到证书。
  - `prologue_extra_fill`：`count < d − 1` 时序言照源码无条件发 `d − 1` 次填充，并证明了多出的填充。
- **`cp.async`（`Async.lean`）。** T1/T2：每线程独立提交，`pending = d − 2`，CTA 会合；WGMMA 路径带 proxy fence 和步内 `wait_group 0`。
- **两个消费 warpgroup 的合作 TMA（`Cooperative.lean`）。** T1/T2：每个 warpgroup 各自 `wait_group 1`，`refilled` 里的 CTA 屏障保证 leader 补填晚于每个组的旧读退役。
- **`ws_fold`（`WarpSpecialized.lean`、`EmptyBarrier.lean`）。** 单 CTA、两个消费组、`lag = 1` 下的 T1/T2：由 empty mbarrier 的计数和奇偶推出八个 warp leader 都已交还旧槽。
- **覆盖。** 11 个 bench 环里，**7 个原则上满足模型前提**：5 个 `cp.async`、F32 的双消费者 `gemm_tiled_wg_tma`、`gemm_tiled_wg_ws`。
  - 剩下 4 个：cluster 2 个、persistent 1 个、epilogue 叠放的 F16 双消费者 1 个。
  - `LowerToMem` 的分类日志还没跟上，仍报告 0 个覆盖；README 写明了分类器应当怎样改。

`lake build` 通过，主要定理只依赖 Lean 标准公理。

### 20.3 下一步

1. 按 README 更新分类器，并把 `cp.async` 环的布局字段打进日志。
2. 面板句柄与 `rid` 绑定，引入线性 `RingState rid`，防止同槽重复 `await`。
3. 用 Lean 的 `Code` 证书检查真实的残余 fold，而不只是合成夹具。
4. cluster、persistent、叠放的模型扩展。

## 21. 第三轮（2026-11-19）

### 21.1 面板句柄（`961c8c9a6`）

`gpu.Panels (rid, depth) P` 是环槽的类型化句柄，只能由私有的 `gpu.panels` 构造。`read_slot` 只接受同一 `(rid, depth)` 的句柄和令牌，因此拿另一个环的面板去读会在类型层被拒（`lit/error/ring_panels_other_ring.mim`）。`ring_fold_at` 改为对 `Held (rid, depth)` 特化。`gpu.erase_phases` 把句柄擦成原来的数组。全量 lit 通过，73 个 benchmark 的设备 IR 经 `canon_ir.py` 后逐字节不变。

### 21.2 分类器（§20.3 第 1 项）

`-X tensor:ring-coverage=1` 的日志给出每个环的 `model` 和 `groups`，`ring_coverage.py` 结合 NVPTX 的放置结果给出结论：

| 结论 | 环数 | benchmark |
| --- | ---: | --- |
| `covered`，`async` | 5 | `attn_{law,roll,warp,wg}`、`fusion/attn_warps` |
| `covered`，`cooperative` | 1 | `gemm_tiled_wg_tma` |
| `covered`，`warp_specialized` | 1 | `gemm_tiled_wg_ws` |
| `overlaid`（协议为 cooperative） | 1 | `gemm_tiled_wg_tma_h` |
| `cluster` | 2 | `gemm_tiled_wg_{cluster,split}` |
| `persistent` | 1 | `gemm_tiled_wg_persistent` |

合计 7/11，与 Lean 扩展预计的一致。`lit/gpu/ring_fold_coverage.mim` 现在对三条真实路径各编译一个 benchmark。这仍是“形状落在模型前提内”的判断，不是对残余代码的证书检查。

### 21.3 尚未做的

- **RingState。** 同槽重复 `await` 仍未拒绝。§17 定的“能力 + 残余验证”中，能力一半需要把 `RingState rid` 穿过 `SyncOf` 的每个操作。
- `gpu.in_flight` 已改为 `priv`：只有 `wgmma_batch` 的等待能给出 `InFlight pending` 见证，原先伪造见证的两个测试改为经过 0 条 MMA 的 `wgmma_batch`（`lit/error/ring_in_flight_forge.mim`）。
- **rolled 环与 `ws_fold` 仍直接读 `slots#0`**，没有经过 `read_slot`。它们每步轮转槽数组，第 0 位是物理槽 `block % depth` 这一点是循环不变量。要让句柄类型带上轮转量，`affine.For` 的状态类型就得依赖迭代号，与 RingState 是同一个阻碍（§17）。改为不轮转、按动态下标读，会改变设备 IR，所以暂不做。
- **真实残余的证书检查（第 3 项）。** 单消费者 TMA 有可计算的 `Ring.fold`/`Code`，但 11 个真实环里没有单消费者 TMA。`Async`、`Cooperative`、`WarpSpecialized` 是以事件时间表陈述的 `Prop` 结构，没有可执行的检查器。要检查真实 kernel，需要先为每个模型写可计算的逐参与者事件检查器，并证明它蕴含相应的 `Fold`/`ReadCode`；再从 `gpu.erase_phases` 之前的残余里抽取逐线程（或逐 warpgroup）事件。后者可沿用 `ring_fold_trace` 的主机执行办法：把 `threadIdx` 当参数，枚举代表线程。
- cluster、persistent、叠放的模型（第 4 项）。

### 21.4 真实 kernel 的证书检查（§20.3 第 3 项，cp.async 部分）

- **Lean 检查器**（`391a25ebd`，`Ring/AsyncCode.lean`）：可执行的 `accepts d count proxy P`，检查单个线程的 cp.async 环指令序列。`fold_of_accepted`、`readCode_of_accepted` 证明：每个线程都被接受，且时间表满足程序顺序、CTA 屏障和硬件规则时，就有 `Async.Fold` 和 `ReadCode`，因而 T1/T2 成立。可执行文件 `ring_async_check` 读文本格式。
- **提取器**：`gpu.ring_trace`，默认流水线里紧跟 `check_sync`，只在 `-X gpu:ring-trace=1` 时工作。它是一个 `Rewriter`，借 `Lam::reduce` 代入并折叠算术，按代表线程解释残余：
  - 循环按字面量边界展开，`gpu.bind` 循环只走第 0 次迭代；
  - `mem.fresh` 换成新变量；
  - `cps2ds_dep` 的调用被执行，结果代回；
  - `ringFill` 记为一次 fill，不进入；
  - `ringStep` 进入，统计其中的 WGMMA 批次。
- **槽的命名**：按 fill 首次写到物理缓冲区的顺序编号。T1/T2 在槽的双射重命名下不变，两张票写进同一缓冲区、读到没被 fill 过的缓冲区，都会被拒。rolled 环直接读 `slots#0` 也能这样识别。
- **结果**：5 个 cp.async 环（`attn_{law,roll,warp,wg}`、`fusion/attn_warps`）的真实事件序列全部被接受。`attn_law` 的序列与规范序列逐行相同。`bench/tools/ring_trace_check.py` 驱动检查，`lit/gpu/ring_trace_async.mim` 是回归测试。
- **可信边界**：
  - `ringFill` 只写它拿到的槽，`ringStep` 只读它拿到的面板。这两条现在由 `check_sync` 的令牌规则部分保证，没有单独检查。
  - 步内 WGMMA 的效果分属几条 `mem.fresh` 链，在这一 phase 看不出先后。提取器只核对每步的 fence、commit、`wait_group 0` 个数相等，批次内的顺序由 `wgmma_batch` 的定义保证。
  - 各线程写入区间不冲突、所有线程执行同一个 `bar.sync`、warpgroup 收敛，这些不在检查范围内。
  - 解释的是 `check_sync` 时的残余，之后的 phase（`mem.add_mem`、`affine.lower_for`、后端）不在范围内。
- **尚未覆盖**：合作 TMA 与 `ws_fold` 还没有可执行检查器；cluster、persistent 没有模型。

### 21.5 RingState 的可行性（调研，`docs/reviews/ringstate_report.md`）

codex 额度用完，由 Claude 子代理调研。结论：
- **可行，但要分层。** 每个角色带一个擦除的两相位线性状态 `Ring rid depth {Ready, Got}`，由 `await` 和 `release` 推动。它在尾部、第一圈入口、两种 lag、rolled 和 `turns` 下都循环不变，不需要依赖迭代号的循环状态，设备 IR 不变。同一步两次 `await` 在核心类型检查里就被拒（探针 `tsdbl.mim`）。别名（同一个旧状态用两次）仍要 `check_sync` 的单次使用规则。
- **相对坐标系的假设只部分成立。** 轮转后每步的局部状态确实一样，但“相对第 0 位 = 物理槽 `n % d`”仍是关于计数器的归纳不变量；屏障地址、奇偶、补填块号都按物理块号算。它要么由检查器内对循环做语法归纳（方案 F），要么交给 Lean 残余验证。
- **排序**：D′（有限相位 typestate）+ E（残余验证）＞ F（检查器内的游标归纳）＞ B（依赖类型的循环组合子）＞ A（存在打包）＞ C（细化类型）。
- **相关工作**：Async Graphene 的环迭代器类型不含游标，是相对坐标系的先例，但没有线性；CUTLASS `PipelineState` 在运行时持有 (index, phase)；Kuiper 按绝对迭代号索引屏障令牌；Tristan & Leroy（POPL'10）逐实例验证软件流水，是残余验证的先例。
- **另两点发现**：`ws_fold` 拿到 `InFlight` 见证却丢弃，没接到 `release`；`ws_fold` 依赖 `persistent_fill i ≡ i (mod d)`（`tensor.mim:497`），目前无检查。

### 21.6 两相环状态（§21.5 的 D′）

- `gpu.Ring (rid, depth) b` 是每个读者一侧的线性状态，只由私有的 `ring_idle`、`ring_got`、`ring_put` 产生和推进，`ring_got`/`ring_put` 都带 mem，免得哈希共享把两次推进合并。
- `SyncOf` 多一个参数 `R: Bool → *`：
  - `await: [m, n, R ff] → [mem, H (n % d, tt), R tt]`；
  - `release: [mem, s, live, H (s, live), got: Bool, R got] → [mem, R ff]`；
  - `idle: [H (0, ff), R ff]`。
  - `release` 对相位泛型：`ws_fold` 在 `lag 1` 收尾时于 `R ff` 下交还最后一个槽。严格要求 `R tt` 会多出一个收尾操作和一个字段。
- `ring_fold_at`（rolled 与 `turns`）和 `ws_fold_at` 把状态放进 `Last`，`refilled` 返回 `[mem, R ff]`。`Last` 的分量起了名字，否则擦除后的两个单位分量在 dump 时同名。
- `check_sync` 把 `Ring` 当作令牌，单次使用；`erase_phases` 把它擦成 `[]`。
- **拒绝**：同一状态上连续两次 `await`，前端类型检查即拒（`lit/error/ring_await_twice.mim`）；`release` 返回的状态被用两次，`check_sync` 拒（`lit/error/ring_state_alias.mim`）。
- **缺口**：初始状态 `ring_idle` 与 `unheld` 一样不计使用次数，所以从初始状态出发的两次 `await` 不被拒。要关掉它，需要让 `ring_idle` 带 mem 并只由 `open` 产生。
- 20 个手写协议测试按新接口移植。全量 lit 通过。73 个 benchmark 中 71 个设备 IR 经 `canon_ir.py` 后逐字节相同；`gemm_tiled_wg_{ws,persistent}` 只差一个 LLVM 值名（`%_552.p` 变成 `%m_552.p`），指令相同。
