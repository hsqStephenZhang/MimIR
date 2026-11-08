# Copilot review：tiling 各步的形式化程度（2026-11-08）

由 GitHub Copilot CLI 对分支 `feat/gpu-tiling` 的 review 原文，提示词要求以"每一步 tiling 是否 well-formed/well-typed"为视角、对照 `cgo27-paper248.pdf`（它无权读取 PDF，按提示词里的 Tileable/retile 模型对照）。未经修改。

# Tiling 的形式化与类型完整性 review

**结论：当前类型保证的是“部分形状关系和 levels 的作用域语法”，不是“每一步 tiling 都合法且保义”。** 分块覆盖、归约顺序、共享视图包含关系及 collective 的一致到达仍有缺口，其中树形合并存在可直接从代码推导的漏算情形。

**基线限制：**当前权限拒绝读取指定 PDF，因此未能核验论文原文。以下以你列出的论文模型，以及工作树保留的 `Tileable/retile` 实现（`src/mim/plug/tensor/tensor.mim:644–655、727–740、882`）作对照；不把代码中的说明当作已核验的论文证明。全程只读，未修改文件。

## 1. 每一步实际上由什么保证

“类型保证”在这里指调用接口本身排除非法组合；C++ normalizer 内的检查仍属于可信实现，不等于不变量进入了类型。

| 步骤 | 当前保证及缺口 | 代码位置 |
|---|---|---|
| **op → op retile**：`tile_of`、`tile_of_box` | **类型保证：**输入/输出元素类型、rank、输出 shape=`ext`。**C++ 分析：**canonical schedule、write preimage exact、输入 image、storage cap；`ext≤cap` 仅在两者都是字面量时显式检查。**完全没有：**接口中的合法 box 证据。`tile_legal` 判断的是是否成功生成局部 op，不是完整的 box 合法性证明。 | `src/mim/plug/tensor/tensor.mim:1524–1554`；`src/mim/plug/tensor/normalizers.cpp:337–414、551–554` |
| **归约分块及 post**：init → partial fold → post | **类型保证：**同 shape 的 accumulator 流转、post 输出类型。**C++ 分析：**实际 `To/Tp` 匹配、loop 属于归约维、post 的语法恒等判定。**用户证明义务，但未要求提供证据：**chunk 覆盖、顺序、必要的代数律。接口没有约束 `k0/k` 与原归约域的关系。 | `src/mim/plug/tensor/tensor.mim:1555–1573`；`src/mim/plug/tensor/normalizers.cpp:380–388、517–589` |
| **levels → region 嵌套** | **类型保证：**三个链的作用域转移、sizes 长度共用 `n`、merge 使用链的 `Acc`。**C++ 分析：**`n` 与 root 循环数相等、维度角色、literal sizes、tail、单归约维、Vec 必须最内层、multi-result axes 的唯一性和 extent 匹配。body 类型只保证返回形状，不保证输入访问合法、输出 ownership 或执行作用域。 | `src/mim/plug/tensor/tensor.mim:1746–1766、1801–1802、1955–2015`；`src/mim/plug/tensor/phase/tile_fuse.cpp:899–949、1140–1147、1273–1284`；`src/mim/plug/tensor/normalizers.cpp:632–667` |
| **融合及 resident 复用** | **C++ 分析：**provenance、outside uses、yield 覆盖、循环依赖排除、可局部化 producer。members 限制实际由 `default_control` 策略执行，而非不可绕过的类型约束。**完全没有：**resident 子视图包含于已生成共享 tile 的证明；不同 box 时直接减 offset、构造 `slice_box`。 | `src/mim/plug/tensor/phase/tile_fuse.cpp:754–770、829–874、1018–1079`；`src/mim/plug/tensor/tensor.mim:1819–1824` |
| **内存层级及 bufferization** | **类型保证（低层局部）：**现有 `GlobalM/SharedM/LocalM`、指针地址空间确实有类型。**C++ 分析：**uniform 推断、cap 上界、in-place 使用、view/proxy 重建。**完全没有：**tensor/region 类型中的存储 kind、group/item owner、生命周期及初始化范围；generic pointer/token 隐藏了这些区别。 | `src/mim/plug/tensor/phase/lower_to_mem.cpp:260–269、531–548、2267–2386`；`src/mim/plug/gpu/gpu.mim:33–43、254–273` |
| **跨 item 归约 R8** | **类型保证：**merge 的输入/输出类型。**C++ 分析：**block count、直接 group 环境、in-place step、静态 storage、combiner 结构、power-of-two 分派。**用户证明义务：**seed 单位元、merge 代数律、merge 与 step 的兼容性；代码不验证。**完全没有：**树形 pair 分配完整覆盖的证据，而且当前分配会漏 pair。 | `src/mim/plug/tensor/phase/tile_fuse.cpp:785–798`；`src/mim/plug/tensor/phase/lower_to_mem.cpp:2527–2596、2700–2748` |
| **执行映射及 barrier** | **类型保证（launch 边界）：**kernel 的 `Idx n_groups`、`Idx n_items` 和地址空间 token。**C++ 分析：**绑定循环静态 trip count、最大 item 数、scope 检查、短循环守卫。**完全没有：**region 与 grid/block 的类型联系、barrier 的一致到达和跨转换保持；目前依赖 continuation 参数化及禁止内联。 | `src/mim/plug/tensor/phase/lower_to_mem.cpp:1909–1954、2669`；`src/mim/plug/gpu/phase/outline.cpp:31–40、84–124、176–180、228–266` |

## 2. 按优先级排序的改进建议

### 1. P0：先修复树形合并的覆盖条件，不要把“item 数是 2 的幂”当作充分条件

**要捕获的不变量：**每一轮的全部 `s·elems` 个 pair 恰好合并一次。

当前计算为：

```text
pairs = s·elems
trip  = min(n, pairs)
per   = pairs / trip
pair  = t + q·trip
```

见 `src/mim/plug/tensor/phase/lower_to_mem.cpp:2703–2714`。没有余数处理。

例如 **`n=8、elems=3、s=4`**：需要处理 12 个 pair，但 `trip=8、per=1`，只处理编号 0–7，编号 8–11 被遗漏。`n` 已经是 2 的幂；合法的三元素输出 tile 仍会漏算。这是覆盖错误，不是浮点 reassociation 问题。

**建议编码：**将 pair 分配做成明确的 axiom/IR builder：每个 item 遍历 `t, t+trip, … < pairs`，或使用向上取整并守卫 `pair<pairs`。保留完整 batch 快路径时，必须有可归一化的 `pairs % trip == 0` 条件；不要额外限制输出 tile 必须为 2 的幂来掩盖问题。

**代价与风险：**低；主要风险是守卫与下一轮 fence 的位置。增加 `n=8、输出 tile=3` 的 device 路径用例，而不只是已有 CPU sequential fallback 用例。

**与论文关系：**这是多级 lowering 的覆盖证明，属于推广后的必要义务；`retile` 的结果类型无法替这段 C++ 下标分配证明覆盖。

### 2. P0：把“保持顺序的分块”和“可重组的并行归约”分成两个契约

**要捕获的不变量：**

- R3 必须保持原来的逐输出 fold 序列。
- R8 必须满足 partial fold 的组合定律，而不只是 merge 类型正确。

**R3 的具体缺口：**`Levels::reduction` 允许切任意一个归约维，只禁止一次切多个维（`src/mim/plug/tensor/phase/tile_fuse.cpp:1140–1147`）。chunk fold 被放到剩余循环外，而 `producer_tile` 也不要求 cut 是最外层归约维（`src/mim/plug/tensor/normalizers.cpp:380–388`）。

对于两个归约维 `(a:2,b:4)`，切 `b` 为长度 2 的 chunks：

```text
原顺序：00,01,02,03,10,11,12,13
分块后：00,01,10,11,02,03,12,13
```

经典 nest 确实按归约维顺序嵌套（`src/mim/plug/btensor/btensor.mim:399–405`）。因此“chunks 按原顺序 fold”并不足以保证**元素**按原顺序 fold。用 `acc*10+x` 和元素 1–8，即可区分 `12345678` 与 `12563478`。

**R8 的具体缺口：**`merge_of` 只识别 `op(acc,e)` 的结构；scalar seed 被当成单位元，没有验证（`src/mim/plug/tensor/phase/lower_to_mem.cpp:2527–2547、2578–2590`）。整数加法以 `1` 为 seed、拆成四个 partial，原结果为 `1+Σx`，树形结果为 `4+Σx`。此路径也没有检查实际 math mode 是否允许重组。

**建议编码：**

- `OrderedFold`：短期只允许切最外层归约维；推广时显式携带 lexicographic prefix，把 chunk fold 放在正确位置。
- `ParallelFold`：要求 `ReductionLaw(step, seed, merge, mode)`，至少表达  
  `fold(seed, X++Y) ≈ merge(fold(seed,X), fold(seed,Y))`。
- 当前 halving tree 会重排 partial，需交换律；若改成相邻区间的保序树，可减少这项义务。
- 内置运算由受信任 normalizer/rule 根据运算、seed 和 mode 发放 witness；用户 merge 必须显式承担证明义务。**`fast` 是数值语义许可，不是任意 merge 的代数证明。**

仅验证 merge 自身构成幺半群仍不够：例如整数 sum 的 partial 用 XOR 合并，类型、结合律、交换律和零单位元都成立，但不满足上述兼容定律。

**代价与风险：**R3 限制成本低；完整 law 接口成本中高，风险在于把“许可误差”错误包装为精确等价证明。

**与论文关系：**保序 R3 是 retile 模型的推广；R8 则引入新的归约语义契约，不能只解释为身份调度标注。

### 3. P0：让 levels 索引真实的 root descriptor，而不是 root 的输出元素类型

**要捕获的不变量：**`n` 是该 root 的循环数，`Acc` 是它的 accumulator 类型，merge 正好作用于这个 accumulator。

当前 `tile_req` 的：

```text
op: Tensor r s T
levels: TileLevels(n,T)
```

见 `src/mim/plug/tensor/tensor.mim:1801–1802`。但 `map_reduce_post` 输出的是 **`Tp`**，accumulator 是 **`To`**（同文件 `167–177`）。

因此，`To≠Tp` 时：

- 当前 levels 把 `Acc` 绑定到 `Tp`；
- 生成 `merge_with` 时却要求 `mr.To`（`src/mim/plug/tensor/phase/tile_fuse.cpp:1237`）；
- 正确的 `To` merge 会在入口被拒绝，错误针对 `Tp` 的 merge 则可能通过入口，直到构造内部 IR 才失败。

现有负例只覆盖 `To=Tp=F32`，不能证明一般情况（`lit/tensor/tile_levels_merge.mim:18–21`）。

**建议编码：**引入类似 `ReductionTileable` 的 descriptor，显式打包 `Acc、Elem、Domain、Out` 和 init/partial/post 能力；`tile_req` 接收 `Levels(descriptor.Domain, descriptor.Acc)`。把输出 Tensor 留作 descriptor 的 materialization，而不是反向从 Tensor 猜 accumulator。

同时让 Domain 描述 parallel/reduction 维度角色；将 `ParSizes`、单轴 `RedSplit`、末端 `Vec` 分开，避免全长 Nat 向量表达所有级。

**代价与风险：**中；会迁移构造 helper 和调用点，但能消除真实的接口不一致。literal sizes 仍可作为后端支持限制检查，不必声称纯 Nat 类型能证明“语法上是字面量”。

**与论文关系：**这是直接继承 Tileable 的“隐藏实现、显式暴露变换能力”原则，并补充归约能力；比裸 Tensor 加 C++ 反射更接近基线。

### 4. P1：把 box 合法性与视图包含关系做成可携带、不可随意伪造的证据

**要捕获的不变量：**

- `off≤shape`，`ext≤shape−off`；
- `ext≤cap`；
- resident 子视图完全位于已生成 tile 的**逻辑有效范围**；
- storage layout 足以容纳映射后的访问。

注意：**不应要求 `off+cap≤shape`**；尾 tile 的物理 capacity 可以超过逻辑有效 extent。

当前 `slice_box` 的类型只有返回 shape=`ext`，reference 实现忽略 cap（`src/mim/plug/tensor/tensor.mim:1531–1544`）。resident 复用只检查同 box，其他情况直接构造相对 offset 的视图（`src/mim/plug/tensor/phase/tile_fuse.cpp:829–839`）。

另外，`preimage.exact` 不能替代 box 合法性：它描述裁剪到原 domain 后的 preimage；实现明确用原 extent 截断范围（`src/mim/plug/affine/normalizers.cpp:680–686`）。

**建议编码：**

- `Box(shape)`、`CappedBox(shape,cap)`：由受控构造子和归一化 Bool 条件产生合法性 witness。
- `Subview(parent,box)`：包含 witness 必须以 **parent 的有效 ext** 为基准，不能只比较 storage cap。
- `resident_` 保存 producer、box、scope owner 和证据；替换前消费包含证据。
- 静态不能证明时，保持原 producer 读取，或显式生成运行时边界分支；不得把“未知”当“包含”。

`affine.image` 作为输入 read 的保守 hull 不必处处 exact；真正需要证明的是访问被 hull 包含、以及局部结果完整覆盖目标输出。

**代价与风险：**中；主要风险是符号算术不归一导致优化减少。采用“证明成功才应用”的 rule，不需要先引入通用定理证明器。

**与论文关系：**这是把 iota/omega 的维数关系推广成受限 domain 间的合法映射；当前裸 offset/cap 注解不足以承担这一推广。

### 5. P1：作用域要进入 region，而不能在 flatten levels 后重新靠标注恢复

**要捕获的不变量：**类型化链的 scope 转移在 region 构造、简化和 lowering 中持续成立。

当前有一个具体破口：`parallel` 在整块覆盖且 `depth>0` 时跳过该 level，**没有排除绑定级**（`src/mim/plug/tensor/phase/tile_fuse.cpp:945–949`）。

例如外层 serial par 后接“整块”的 `serial_group`，再接 `group_item`：链本身合法，但 group region 可能被消去，留下孤立 item 绑定，随后被 outline 拒绝（`src/mim/plug/gpu/phase/outline.cpp:84–90`）。“不改变 tile 大小”并不意味着“不改变执行作用域”；promote 也不能因此丢失。

**建议编码：**region 使用 `Region(scope, domain, exec, …)`，而不是返回类型为任意 `T` 的 `bind`。Group/Item 构造子显式引入下一作用域；整块简化只能消去尺寸切分，必须保留作用域和存储动作。

`exec` 可以携带 grid/block 形状及 linearize/delinearize 关系；launch 和各 item 子循环从同一 descriptor 推导 count，避免分别计算后再取最大值。

**代价与风险：**先保留绑定级成本低；完整 scoped region 成本中高。迁移期可保留 annotation，但增加“flatten 前后 scope 对应”验证器。

**与论文关系：**这是 region-based multi-level scheduling 的推广。作用域变化不是纯 retile 可以忽略的 metadata。

### 6. P1：区分“地址空间类型”与“执行归属、生命周期类型”，统一 host/device token 边界

**要捕获的不变量：**host 与 kernel token 不混用，shared buffer 只属于创建它的 group，private buffer 只属于对应 item，view 不逃出 owner 生命周期。

当前不是“内存 token 完全没有类型”：`GlobalM=mem.M 1` 等已经存在，copy 和 launch 接口也区分它们（`src/mim/plug/gpu/gpu.mim:33–36、154–164、210–218`）。真正缺失的是**类型与执行位置的联系**：

- kernel body 又使用 `mem.M 0`；
- global token 经 `generic_mem/global_mem` 来回转换；
- device pointer 被 bitcast 成原 host pointer 类型；
- shared slot 也返回 `Ptr(T,0)`。

见 `src/mim/plug/gpu/phase/outline.cpp:193–195、239、253–266`；`src/mim/plug/gpu/gpu.mim:260、272–273`。

**建议编码：**先以现有 nom 和依赖参数表示 `Host`、`Device(context)`、`Shared(group)`、`Private(item)`；buffer/view 类型携带 owner、kind、logical shape、capacity。地址空间转换和 token 桥接只能发生在明确的 kernel 边界 rule 中。将两条 GPU host glue 路径统一为同一 session/token descriptor。

`stage.scope: Nat` 不应直接升级为任意数字的内存 kind：当前 reference 中 scope 没有存储语义，实际 placement 来自 `uniform_`（`src/mim/plug/tensor/tensor.mim:2293–2317`；`src/mim/plug/tensor/phase/lower_to_mem.cpp:531–548`）。

**代价与风险：**高，涉及 bufferization/outline；但只加 owner/kind 参数不要求新的 universe kind。若要禁止 token 复制、证明生命周期或独占更新，则需要 effect/ownership 检查，普通 nominal 参数本身不够。

**与论文关系：**这已跨出纯 Tileable 模型；是多级设计的扩展，不能沿用形状类型的证明替代内存归属证明。

### 7. P1：barrier 需要 collective 契约，“位于 group 内”只是必要条件

**要捕获的不变量：**同组所有 item 以一致顺序到达 collective，且优化不会把它复制或下沉到非一致分支。

`gpu.fence: mem.M 0→mem.M 0` 不要求 group capability（`src/mim/plug/gpu/gpu.mim:254`）；outline 只检查是否在 kernel 中（`src/mim/plug/gpu/phase/outline.cpp:117–119`）。短循环靠 state 参数化维持同步位置，merge join 靠 `filter=false` 禁止内联（同文件 `104–115`；`src/mim/plug/tensor/phase/lower_to_mem.cpp:2669`）。

这些是有效的实现补救，但不是 collective 保持定理。

**建议编码：**

- scoped group capability 排除 host/item 外部误用；
- 再增加 collective effect 或 uniform-control 验证：group capability **本身不能阻止在 divergent 分支中调用 barrier**；
- 短 item 循环只守卫工作，不守卫进入下一阶段的 collective；
- 短期在 outline 后及相关内联/CFG 改写后检查 barrier 的一致可达性，禁止不满足规则的转换。

**代价与风险：**中高；通用 uniformity 分析困难，可先限定结构化 region。只用 `if_static` 无法证明由 `item_id` 控制的运行时一致性。

**与论文关系：**这是从纯函数 scheduling 扩展到 SIMT collective effect 的新义务，而非 retile 的结果类型关系。

### 8. P2：建立统一的“规格—合法性—实现”接口，明确 normalizer 的可信边界

**要捕获的不变量：**每次 rewrite 有清晰的规格、前提和残留行为，而不是把“成功构造同类型节点”视为合法性证明。

现有 `tile-ref` 是很好的差分规格入口（`src/mim/plug/tensor/normalizers.cpp:494–514`），但不覆盖 R8 的并行语义契约。`tile_legal` 的 false 还混合了“不支持此 map/schedule”和真正的非法情况。

**建议编码：**

- 区分 `ValidBox`、`Localizable`、`OrderedChunkable`、`ParallelMergeable`。
- C++ analysis 返回带理由的证明结果；rule 仅在对应 Bool 归一到 true 时应用。
- `post` 保留当前 `nps=0`、语法恒等和类型匹配的保守 fast path（`src/mim/plug/tensor/normalizers.cpp:531–543`）；这是 beta/结构恒等，不应声称判断了任意 post 的语义恒等。
- `linear_step/nat_bounded_by` 的分析结论形成可携带 witness，让 bufferization 消费，而不是靠旁表隐式传播。

这里现有 `if_static` 的参数是 **Nat**，判断的是“是否字面量”，不是“Bool 是否为真”（`src/mim/plug/tensor/tensor.mim:606`；`src/mim/plug/tensor/normalizers.cpp:937–944`）。未知条件应走明确的保守分支；它不能充当证明构造子。

**代价与风险：**中；风险主要是证明不足导致优化率下降。一般闭包的代数等价不能靠 normalizer 自动解决，必须明确保留用户义务或受信任 axiom。

**与论文关系：**保留 Tileable 的规格化变换思想，同时承认当前 C++ normalizers 是可信计算基的一部分；实现方式已经偏离“全部由 frontend IR 变换”的描述。

## 3. 哪些“well-formed”声称超出了实际保证

- **“merge 类型匹配 accumulator”只对 `To=Tp` 成立。**`tile_req` 绑定的是输出元素类型；这是接口问题，不只是欠缺代数证明。
- **“顺序 chunks 不改变 fold 顺序”对多归约维不成立。**切内层归约维时会发生元素次序变化；`src/mim/plug/tensor/tensor.mim:1547–1550` 的描述缺少前提。
- **“2 的幂 item 数足以执行树形合并”不成立。**还存在 pair batch 整除条件，当前甚至没有检查该条件。
- **“作用域约束全部成为类型错误”仅适用于链的构造语法。**它没有保证生成的 region 保留 scope；whole-level 消去是反例。
- **“剩在 C++ 的都是类型表达不了的”不成立。**文档 `docs/gpu-tiling-design.zh-CN.md:373–375` 漏列 tail、维度角色、单轴 reduction、末端 Vec、promote 输入范围等检查；这些至少可以用依赖 descriptor、变体及静态 witness 表达。
- **“每级 rewrite 在 Mim 中定义，所以每级已形式化”不成立。**`tf_par/tf_reduce` 定义了结果构造的语义，但没有证明 C++ 选择的 box、循环次序和 resident 替换满足其前提。相应声称见 `src/mim/plug/tensor/tensor.mim:2020–2024`。

反过来，也不应把这些缺口全部称为“类型系统不 sound”：**shape well-typed、访问合法、保义、并行执行无 race/deadlock 是不同性质。**当前确实保证了其中一些局部类型关系，但不足以支持整体性的“tiling 变换总是合法”表述。

## 4. 已读到、但尚不确定的点

- **论文原文未获准读取。**无法判断论文的 well-typed 主张是否本来就只限于类型保持；不能据此认定论文作了更强声称。
- **多 kernel 的 token 冲突根因未确定。**文档记录了 attention 的 `M 1/M 0` 冲突，但同时记录 softmax 三 kernel 可工作（`docs/gpu-tiling-design.zh-CN.md:400`）。已能确定桥接依赖 C++，不能仅从这些代码认定是哪次 fresh-mem/session 重写出错。
- **capped R8 的无效 storage 区域是否被安全处理。**private 初始化和 partial 存储遍历 logical extent，tree 却遍历 storage volume（`src/mim/plug/tensor/phase/lower_to_mem.cpp:2633–2649、2670–2672、2703`）；可能读取未初始化 padding，但其是否影响有效输出仍需沿具体尾 tile/backend 行为确认。
- 上述覆盖、顺序及 seed 反例是静态代码和算术推导，未进行编译或 device 执行复现。

