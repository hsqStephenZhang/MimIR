# 线性 `RingState` 是否可行：研究报告

工作区 `~/workspaces/MimIR-types`，分支 `feat/protocol-types`，HEAD `391a25ebd`。只读调研，未改仓库文件；探针在 `docs/reviews/probes/`。

## 0. 结论先行

- **可行，但要分开讲。**
  - 能做到：一个线性、密封、能拒绝同槽两次 `await` 的 `RingState`。它在 MimIR 现有类型论里**可以表达，循环不变**，不需要依赖迭代号的类型，也不改设备 IR。
  - 做法：以游标为参照系的**有限相位 typestate**，而不是按游标计数的状态。
- **单靠这一层，不能把令牌钉在正确的物理槽和代上。**
  - 相对参照系只是把迭代相关的事实挪了位置。下面这些仍按物理块号计算：
    - 屏障地址 `bars[n % depth]` 和奇偶 `parity n`（`gpu.mim:917-921`）；
    - 补填块号 `n + ahead`（`:1063-1066`）。
  - “相对位置 0 就是物理槽 `n % depth`”是关于循环计数器的归纳不变量。三种证法：
    - 依赖类型的循环组合子加取模化简；
    - 检查器做一次语法归纳；
    - 交给 Lean 的残余验证。
- **主开发者的假设：部分成立。**
  - **成立的部分**：
    - rolled 环和 `ws_fold` 在轮转后的坐标系里，每一步的局部状态都一样：第 0 位待等，第 `depth−1` 位是上一步持有的槽，补填也落在第 `depth−1` 位。
    - `turns` 环在每圈边界上，稳态的槽相位向量也一样。
    - 因此携带类型可以不依赖迭代号。
  - **不成立或要补充的部分**：
    1. **尾部。** 按槽的“填充状态向量”在尾部不是不变的：
       - `exact` 环在 `blocks <= block + ahead` 时不补填（`:1068`、`:1105`）；
       - `turns` 环在 `block >= blocks` 时跳过整步（`:1115`）。
       - 两条分支汇合时槽状态不同。向量类型只能用存在类型或“幻影”转换抹平，而幻影转换的安全性又要靠算术推理。
    2. **第一圈入口。** `idle` 的类型是 `H (idx 0, ff)`（`:893`、`:1122`），而稳态边界上持有的是槽 `depth−1`。
    3. **坐标系之间的对应。** 在相对坐标系里，令牌与物理屏障的对应仍然依赖迭代（见上）。
- **由此得到一个更小、更稳的设计。**
  - 每个角色只带一个两相位的线性状态 `Ready ⇄ Got`，由 `await` 和 `release` 推动，不记录填充状态。
  - 它在以下情况下**都是循环不变的**：尾部两条分支、第一圈入口、两种 `lag`、rolled 与 `turns`。
  - 同一步内的第二次 `await` 在**核心类型检查**里就被拒（探针 `tsdbl.mim`，§2.1）。
- **推荐（§4）。**
  - 方案排序：D′+E（擦除的有限相位 typestate + 残余验证）＞ F（检查器内的游标归纳）＞ B（依赖类型循环组合子）＞ A（存在打包）＞ C（细化类型）。
  - 最小下一步：在 `SyncOf` 里加入 `Ring rid depth phase`，串进 `ring_fold_at`/`ws_fold_at` 的循环状态，在 `check_sync` 里登记为单次使用的令牌。

## 1. 现状与阻碍的准确位置

### 1.1 令牌与环操作

- **令牌。** `Held (rid, depth) (s, live)`、`Vacant (rid, depth) s` 在 `gpu.mim:868-876`。构造器 `hold`/`unheld`/`vacate` 为 `priv`。
- **面板。** `Panels (rid, depth) P` 与 `read_slot` 在 `:878-882`。只有 `turns` 环经过 `read_slot`（`:1095`）。
- **`SyncOf depth H V`**（`:889-894`）有七个字段：`open, acquire, commit, await, release, idle, settle`。
  - **`await (m, n)` 不消费任何东西**，只要一个 `mem.M`。
  - 这就是同槽两次 `await` 能得到两个活跃令牌的根源（design §19.2 F5、§21.3）。
- **环构造器。**
  - 共有 `async_ring` `:906`、`mbar_ring` `:916`、`shared_ring` `:928`、`consumer_ring` `:946`、`fenced_ring`/`resumed_ring`/`cluster_ring`/`pingpong_ring` `:965-1008`、`tma_ring` `:1014`。
  - 大多只转发 `sync#k`。真正包裹 `release` 的只有 `shared_ring` 和 `consumer_ring`。

### 1.2 三种循环

| 循环 | 位置 | 读面板 | 令牌的槽 | 交还 | 补填 |
| --- | --- | --- | --- | --- | --- |
| rolled `ring_fold` | `gpu.mim:1049-1080` | `slots#0`，每步左移一位（`:1058`） | 物理 `idx (block % depth)`（`:1059`） | `last` 的槽，即上一步的 | `slots#ahead`，物理 `(block+ahead) % depth`（`:1063-1065`） |
| `turns` `ring_fold` | `:1081-1121` | `read_slot (panels slots, j, idx(block % depth), mine)`（`:1095`），`block = i·depth + j` | 同上 | 同上 | `slots#later`，`later = (j+ahead) % depth`（`:1090`） |
| `ws_fold` | `:1143-1181` | 生产者、消费者都读 `slots#0`，每步左移（`:1152`、`:1155`、`:1157`） | 物理 `idx (nth block % depth)` | `lag 0` 交还本槽，`lag 1` 交还上一步的（`:1161-1162`） | 生产者 `acquire (m, n)`，填 `slots#0` |

关键观察：
- **补填的正是刚交还的槽。** `ahead = depth − 1`（`:1038`）。rolled 环左移后，补填的 `slots#ahead` 是物理槽 `(block + depth − 1) % depth = (block − 1) % depth`，即上一步交还的那个。所以在轮转坐标系里，每步完全一样：
  1. 读第 0 位；
  2. 交还第 `depth−1` 位；
  3. 补填第 `depth−1` 位；
  4. 左移。
- **`ws_fold` 的令牌与面板靠一个算术事实相容。**
  - 令牌用 `nth block`，面板用按 `block` 轮转的 `slots#0`。
  - 二者相容依赖 `persistent_fill (round, count, depth) i ≡ i (mod depth)`（`tensor.mim:497-500`，形如 `(…) * depth + i % depth`）。
  - 这个事实现在没有任何检查覆盖。
- **`lag 1` 的退役见证没有接上 `release`。**
  - `ring_fold` 的 `step` 不返回 `InFlight` 见证（`:1034`）。所以 `lag 1` 下“上一步的 MMA 已退役才交还”在 `ring_fold` 里没有类型。
  - `ws_fold` 的 `step` 返回 `InFlight lag`，但见证被丢弃（`:1157` 的 `_`），也没有与 `release` 相连。

### 1.3 阻碍到底在哪里

- **组合子的状态类型是固定的。**
  - `affine.For` 的状态类型 `«i: n; Ts#i»` 固定（`affine.mim:30-33`）。
  - `vec.fold.l` 也是非依赖的（`vec.mim:29`）；`turns` 环一圈内的 `depth` 步就是用它折叠的（`gpu.mim:1117`）。
  - 所以以计数器为下标的状态（`RingState rid n`、旋转了 `n` 位的面板句柄）都不能放进循环状态。
- **核心本身支持依赖类型的递归循环。** 手写的递归 `lam loop (m, i: Nat, st: RS (0, 2) i)` 把 `st` 推进到 `RS (i+1)` 后可以通过类型检查（探针 `tsdep.mim`，§2.2）。阻碍在 `affine.For` 这个组合子，不在类型论。
- **下标一取模，核心就判不了等。**
  - `RS ((i % 2 + 1) % 2)` 与 `RS ((i + 1) % 2)` 被拒（探针 `tsmod.mim`）。
  - §14.1 第 4 项又决定这类规则不进全局定义等价。

## 2. 探针

全部在 `docs/reviews/probes/`，用 `build-debug/bin/mim` 运行，每次 `ulimit -v 16000000; timeout 600`。文件名要短：插件名由文件名推出，太长会报 `invalid annex name`。

### 2.1 有限相位 typestate 在核心里就拒绝第二次 `await`

`ts_common.mim` 的接口用独立的 axiom，不依赖 `gpu` 插件：

```
axm RS: [rid depth: Nat] → Nat → *;          // 0 = Ready, 1 = Got
axm await:   {rid depth: Nat} → [m: mem.M 0, n: Nat, RS (rid, depth) 0] → [mem.M 0, TK (rid, depth), RS (rid, depth) 1];
axm advance: {rid depth: Nat} → [mem.M 0, TK (rid, depth), RS (rid, depth) 1] → [mem.M 0, RS (rid, depth) 0];
```

| 探针 | 内容 | 结果 |
| --- | --- | --- |
| `tsstep.mim` | 一步：`await` 再 `advance` | 通过 |
| `tsdbl.mim` | 把第一次 `await` 返回的状态再交给 `await` | **前端拒绝**：`expected [core.M 0, Nat, RS (0, 2) 0], got [core.M 0, Nat, RS (0, 2) 1]` |
| `tsalias.mim` | 两次 `await` 都用**同一个**旧状态 `st` | 通过：别名重用只能由线性检查拒绝，即 `check_sync` 的单次使用规则 |

- 相位下标负责“顺序”，线性负责“不复制”，二者合起来才能排除两次 `await`。
- 相位是字面量，判等不涉及 §14.1 第 1 项的自由变量问题。

### 2.2 依赖类型循环在核心里可写，取模下标不行

- **`tsdep.mim`，通过。** `adv: RS n → RS (n + 1)`；递归 `loop (m, i, st: RS (0,2) i)` 调用 `loop (m2, i + 1, st2)`。
- **`tsmod.mim`，被拒。** `adv: RS n → RS ((n + 1) % depth)`，`loop` 的状态为 `RS (i % 2)`。核心不知道 `((i % 2) + 1) % 2 ≡ (i + 1) % 2`。

### 2.3 擦除后的单位分量不进 LLVM IR

- **探针。** `ghost_unit_a.mim` 与 `ghost_unit_b.mim` 是同一个 `affine.For`，后者的循环状态多一个 `[]` 分量。
- **结果。** 用 `-p opt … -p ll` 生成的 `.ll` 除 gid 后缀外**逐行相同**（`ghost_unit_{a,b}.ll`）。
- **与现状一致。**
  - `gpu.erase_phases` 把 `Held`/`Vacant`/`InFlight` 擦成 `[]`（`erase_phases.cpp:12-20`）。
  - 现有的 `Last` 早已把擦除后的单位分量带进循环状态，§18、§21.1 也报告 73 个 benchmark 的设备 IR 不变。
- **局限。** 这只是 CPU 后端的小例子。新增一个擦除分量能否让 73 个 benchmark 逐字节不变，仍须用 `canon_ir.py` 实测。

## 3. 问题 1：各方案在原则上能否表达

记号：
- **静态**：Mim 前端的类型检查，或 PE 之后、擦除之前的 `check_sync`。
- **残余**：从擦除前的残余抽取事件，由 Lean 证书检查器验证。`proofs/ring/README.md` 有说明；`AsyncCode.lean` 已有 `cp.async` 的可执行检查器 `accepts`。

### 3.1 逐条核对主开发者的假设

**(i) `turns`：每圈边界上的槽相位向量相同。**
- **稳态下属实。**
  - 第 `i` 圈开始时，块 `i·d .. i·d+d−2` 已提交到槽 `0 .. d−2`，`last` 持有槽 `d−1`。
  - 第 `j` 步依次是：等槽 `j`（满→持有）；交还槽 `j−1`（持有→空）；补填槽 `(j+d−1) % d = j−1`（空→满）。
  - 所以每步是 `[Full(j), Held(j−1)] → [Held(j), Full(j−1)]`，一圈后回到 `{Full(0..d−2), Held(d−1)}`。
- **三处破坏：**
  1. **第一圈入口。**
     - 槽 `d−1` 从未填充，`last = (ff, idx 0, idle)`（`:1122`）；`idle` 的类型下标是槽 0，不是 `d−1`。
     - 把它改成 `d−1` 会改变 `Last` 的运行时初值 `s`，从而改变设备 IR。只能让幽灵状态独立于运行时的 `s`。
  2. **`exact` 尾部。** `done` 分支不补填（`:1104-1105`）。它与 `refilling` 分支汇合时，槽 `j−1` 一边满、一边空。
  3. **最后一圈的 `skip`**（`:1093`、`:1115`）。状态原样返回，与 `go` 分支在位置上差一步。
- **要让向量不变，需要两个“幻影”转换：**
  - `done` 分支凭空产出 `Full(j−1)`；
  - `skip` 分支把状态快进一步。
- 二者只在守卫 `blocks <= block + ahead`、`block >= blocks` 下安全，也就是“幻影槽以后不会被等”。
  - 这是关于块号的算术蕴涵。
  - `turns` 下还要知道：跳过之后，同圈的后续步也都跳过，循环随即结束（单调性）。
  - 这些只能静态地做语法特判，或者交给残余验证。
- **代不在向量里**，正如假设所说。
  - `Held` 没有代，向量只区分满、空、持有。
  - 读到上一代（parity 算错、少填一块）这一层发现不了。

**(ii) rolled 与 `ws_fold`：在轮转坐标系里，“第 0 位是下一个要等的槽”不变。**
- **属实。**
  - rolled 的状态是 `[Full^{d−1}, Held]`，第 `d−1` 位是上一步的槽。每步“等第 0 位、交还第 `d−1` 位、补填第 `d−1` 位、左移”之后回到同一形状。
  - `ws_fold` 的消费者：`lag 0` 下每步等第 0 位、交还第 0 位；`lag 1` 下等第 0 位、交还第 `d−1` 位。两者都不变。
- **但若令牌与面板都用相对下标，物理一侧就失去了类型：**
  - `await (m, n)` 等的仍是物理屏障 `bars[n % d]` 和奇偶 `parity n`（`:917-921`）；
  - `release` 用物理的 `s` 计算 `buf_addr (empties, s)`（`:935`、`:957`）。
  - “相对第 0 位 = 物理 `n % d`”就是 `rot ≡ block (mod d)`，它依赖迭代。
  - 相对坐标系让 `read_slot` 的**位置**可以静态检查，代价是**屏障**与面板的对应不再有类型。
- **反过来也一样。** 用物理令牌加相对面板（现在 `turns` 和 rolled 的混合做法）时，令牌对得上屏障，面板对得上位置，两者之间还是那条不变量。
- **结论。**
  - 这个假设把阻碍从“携带类型”移到了“坐标系之间的对应”，没有消除它。
  - 这条对应的形式很简单：初值 0、每步加 1、`For` 从 0 开始、步长 1。因此适合用检查器做一次语法归纳（§3.2 方案 F），或者交给残余验证。

**(iii) 两次 `await` 是否被拒。**
- **同一步内：被拒。** `RingState` 无论是按位置的状态向量还是两相位状态，只要 `await` 消费它：
  - 第二次 `await` 在核心类型检查里就失败（探针 `tsdbl`）；
  - 复用旧状态值由单次使用规则拒绝，`tsalias` 说明这条规则必不可少。
- **跨步“重复等同一个块”：不被拒。** 例如编译器 bug 让第 `b` 步等 `b−1`：每步一次 `await` 的形状是对的，错在传入的 `n`。这一类要靠游标对应（§3.2 方案 F 或残余验证）。
- **hash-consing 的影响（design §4.4）。** 参数完全相同的两次 `await` 会合并为一个 Def，相当于只等一次，语义上无害。

**(iv) `lag 1` 交还上一个槽、补填第 `ahead` 个槽。**
- **一个相位机器同时适用于两种 lag。** `ring_fold` 在两种 lag 下的环操作顺序相同，都是 `await → release(last) → acquire → commit`，只有 `step` 的位置不同（`:1070-1076`）。只由 `await`/`release` 推动的相位机器因此两种都适用。
- **补填落在刚交还的槽。**
  - 在相对坐标系里，它是“第 `d−1` 位”，不变。
  - 在 `turns` 里，它是字面量 `j−1`。`(i·d + j + d − 1) % d` 由 `check_sync` 已有的 `same_slot`/`nat_form` 同余比较处理（`check_sync.cpp:96-142`）。
- **退役见证（加强项，不是 `RingState` 的必要部分）。**
  - 交还上一步的槽时，“它的 MMA 已退役”需要 `InFlight lag` 见证；`ring_fold` 没有这个见证，`ws_fold` 把它丢弃了。
  - 可以顺带收进相位机器：`lag 1` 时，`release` 要求本步 `wgmma_batch` 给出的 `InFlight 1`。

**(v) 擦除与成本。**
- **擦除。** 与 `Held` 一样擦成 `[]`，转换擦成 `mem` 的直通（仿照 `erase_phases.cpp:17-20`）。探针 §2.3 支持“多一个单位分量不进 IR”，但须在 73 个 benchmark 上复核。
- **`check_sync`。** 把 `Ring` 登记进 `is_slot`/`has_slot`（`check_sync.cpp:45-54`）。现有的单次使用、闭包、循环重复运行、刚性比较规则会自动覆盖它，估计几十行。幻影转换或游标归纳另计（§3.2）。
- **`gpu.mim`。**
  - `SyncOf` 的相关字段多一个状态参数；
  - 两个包裹 `release` 的构造器要转发状态；
  - `ring_fold_at`/`ws_fold_at` 的 `S` 多一个分量；
  - `ws_done` 末尾那次额外的 `release`（`:1179`）改成一个消费状态的收尾操作。
- **`LowerToMem`。** 它只构造 `tma_ring`/`async_ring` 和 fold，类型变化不涉及它的调用形状（`lower_to_mem_stage.cpp:754-784`）。

### 3.2 各方案

| 方案 | MimIR 核心需要什么 | 设备 IR | 静态检查 | 残余验证 |
| --- | --- | --- | --- | --- |
| **D′ 有限相位、相对坐标的线性 typestate**（推荐的主体） | 不改核心；`Ring rid depth φ`，`φ` 为字面量相位；`check_sync` 登记单次使用 | 擦成 `[]`，应不变（§2.3，待在 73 个 bench 上复核） | 每步恰一次 `await`、一次 `release`，且 `await` 在前；不复制、不伪造（私有构造器 + 单次使用）；`turns` 下读槽同余 | 传入的 `n` 是游标；代/奇偶；尾部的补填与跳过；跨角色的满/空交接 |
| **E 残余验证承担游标算术** | 无 | 不变 | 只有局部纪律 | Lean `Code` 风格的逐线程检查器：票号 = 已提交组数，槽 = `j % d`，源块正确（`AsyncCode.lean` 的 `accepts` 已对 `cp.async` 这样做） |
| **F 检查器内的游标归纳**（D′ 的加强，推测） | 不改核心；`check_sync` 识别携带 `Ring`/`Panels` 的 `affine.For` 并核对三条：初值游标为 0 且面板未轮转；每条到达 `yield` 的路径恰好轮转一次；`await`/`acquire` 的 `n` 是 `bitcast i` 加静态偏移 | 不变 | D′ 的全部，加上：相对第 0 位 = 物理 `n % d`；令牌跨步钉在正确的槽 | 代/奇偶；尾部 |
| **B 依赖类型循环组合子** `dfor: Π i. S i → …` | 新组合子及其降级，须与 `affine.lower_for` 生成同样的循环；取模化简（§2.2 失败处），按 §14.1 只能作为协议内引理 | 风险高：手写循环的降级、`ll.nounroll` 等注解须逐字节一致 | 游标、槽、轮转全部进类型 | 代（除非 `n / d` 也进类型）；硬件语义 |
| **A `Σ n. RingState rid n`** | 依赖 Sigma 已有 | 不变 | 只有存在打包内部的局部一致，`n` 与计数器无关 | 与 E 相同 |
| **C 用单例或细化类型连接计数器与下标** | 细化类型或 SMT，MimIR 都没有 | — | 原则上最强 | — |

排序与理由：
1. **D′ + E。**
   - 优点：今天就能做；核心不动，设备 IR 不变；直接关掉审阅给出的“同槽两次 `await`”反例；与 §17 已定的“能力 + 残余验证”一致。
   - 弱点：游标对应完全靠残余验证，而 11 个真实环里只有单消费者 TMA 有可执行证书（design §21.3）。
2. **D′ + F + E。**
   - F 是有界的语法归纳，相当于把 B 的循环原则只对 `ring_fold`/`ws_fold` 这几个固定形状实例化。
   - 它把“令牌钉在正确的槽上”从残余验证移到静态检查，又不需要核心的取模规则。
   - 代仍交给残余验证。
3. **B。**
   - 类型论上最干净（`tsdep` 说明核心能表达），论文叙述最强。
   - 但取模判等和循环降级两个风险都在关键路径上。
   - 建议作为后续工作，或者在论文里作为“原则上”的讨论。
4. **A。** 现在的 `Last = [live, s, H (s, live)]`（`:1036`）已经是 `Σ s live. H (s, live)`，再套一层 `Σ n` 不带来新的保证。
5. **C。** MimIR 没有细化类型，代价不相称。

**对问题 1 的直接回答：能，但按要求分层。**
- **只要求“排除两次 `await`”以及“线性、密封”：** D′ 在现有类型论中可以表达，且循环不变。
- **还要求“令牌钉在正确的槽”：** 需要 F（静态，在检查器内）或 B（静态，在核心内）。
- **还要求“钉在正确的代”：** 在“设备 IR 不变”的约束下，静态层最多做到 F 的游标对应，加上对 `full_parity`/`empty_parity` 的语法核对。奇偶函数本身是否正确，仍属于 `ring_safe`（`tensor.mim:513-524`）和 Lean 定理的范围。

## 4. 问题 3：推荐

### 4.1 论文里最站得住的设计

总体是“**类型化的局部纪律 + 证书化的全局计数**”，分三层：

- **类型层**（核心 + `check_sync`，在擦除前）：
  - 已做：环身份 `rid`、面板句柄、令牌的私有构造与单次使用。
  - 本报告推荐新增：两相位的线性 `Ring` 状态。它在轮转坐标系中循环不变。
- **检查器内的语法归纳**（F，可选的第二步）：把轮转坐标系与物理槽对齐。
- **Lean 残余验证**：游标、票号、代、奇偶、尾部补填、跨角色交接，以及硬件语义（mbarrier、TMA、WGMMA 组）。

叙述上可以与三项相关工作对照：
- **Async Graphene**：迭代器类型不含游标，但没有线性，也不拒绝两次 `get`。
- **CUTLASS `PipelineState`**：游标与相位都在运行时。
- **Kuiper**：屏障令牌按绝对迭代号索引。

我们的定位：把 `PipelineState` 的 index 留在相对坐标系里作为擦除的类型，把 phase 交给证书，设备上不多带寄存器。“逐实例验证、不证明生成器”有 Tristan 与 Leroy 的软件流水验证器作先例。

### 4.2 最小下一步（文件与改动）

1. **`src/mim/plug/gpu/gpu.mim`**
   - 新增 `axm Ring: [rid depth: Nat] → Nat → *;`，相位 0 为 `Ready`，1 为 `Got`。
   - 新增私有转换 `ring_open`、`ring_got`、`ring_ready`。每个都带一个 `mem.M 0`，遵守 §4.4 的约束。
   - `SyncOf`（`:889`）的字段改为：
     - `open: mem.M 0 → [mem.M 0, R 0]`；
     - `await: [m, n, R 0] → [mem.M 0, H …, R 1]`；
     - `release: [mem.M 0, s, live, H (s, live), R 1] → [mem.M 0, R 0]`；
     - `settle: [mem.M 0, R 0] → mem.M 0`。
     - `acquire`/`commit` 不动，生产者一侧由 `Vacant` 的线性管住。
   - `shared_ring`、`consumer_ring` 的 `release` 包裹要转发状态。其余构造器只转发 `sync#k`，类型随之变化。
   - `ring_fold_at`、`ws_fold_at`：
     - `S` 加一个 `Ring … 0` 分量；
     - `ws_done` 在 `lag ≠ 0` 时交还最后一个槽的那次 `release`（`:1179`），改由一个同时消费 `R 0` 和最后令牌的收尾操作完成。这也落实了 §4.2 的“`settle` 显式消耗最后的令牌”。
2. **`src/mim/plug/gpu/phase/check_sync.cpp`**：让 `is_slot`/`has_slot`（`:45-54`）认出 `Ring`。相位比较是字面量比较，`test_alpha` 已经覆盖。
3. **`src/mim/plug/gpu/phase/erase_phases.cpp`**：`Ring` 擦成 `[]`，三个转换擦成 `mem` 直通。
4. **测试**：
   - `lit/error/ring_await_twice.mim`：同一步两次 `await`，前端报类型错误。
   - `lit/error/ring_state_alias.mim`：同一个 `Ring` 值用两次，`check_sync` 报错。
   - `lit/error/ring_release_before_await.mim`。
   - `lit/gpu/ring_state_turns.mim`：覆盖 `exact` 尾部与 `skip`。
5. **验收**：
   - 全量 lit 通过；
   - 73 个 benchmark 的设备 IR 经 `canon_ir.py` 逐字节不变；
   - 报告 PE 节点数和编译耗时的变化（§14.3 的要求）。

**第二步（F）。**
- 在 `check_sync.cpp` 里，对携带 `Panels` 的 `affine.For` 做三条语法核对。
- 然后把 rolled 环与 `ws_fold` 的 `slots#0` 改成经 `read_slot` 的相对读取（`gpu.mim:1071`、`:1074`、`:1155`、`:1157`），关掉 §21.3 的第三项。

### 4.3 保证的分工

| 性质 | 类型 / `check_sync` | Lean 残余 |
| --- | --- | --- |
| 令牌不跨环、不伪造、不复制 | 是（已做） | — |
| 同一步不两次 `await`、不两次 `release`、`release` 不早于 `await`、每步不漏 `release` | 是（D′） | — |
| 读的面板位置与令牌的槽一致 | `turns`：是（同余）；rolled/`ws_fold`：F 之后是 | F 之前由残余覆盖 |
| `await` 的块号是本步的块，补填块号为 `n + ahead` | F 之后是 | F 之前由残余覆盖 |
| 代与奇偶正确：读到本代，覆盖发生在上一代最后一次读之后 | 否 | 是（T1/T2） |
| `exact` 尾部不读未填的槽，`skip` 之后不再读 | 否 | 是 |
| WS 的满/空交接，cluster、persistent 的 `nth` | 否 | 是（部分模型已有，cluster/persistent 仍缺） |
| `lag 1` 交还时上一步的 MMA 已退役 | 可选：把 `InFlight` 见证接到 `release` | 是 |

## 5. 问题 2：相关工作

标记：**[V]** 本次对照原文或权威页面核实过；**[M]** 确认存在，细节凭记忆；**[U]** 未能核实。

### 5.1 GPU 方向

- **Async Graphene** [V]
  - B. Hagedorn, V. Grover, “It's about Time: Temporal Abstractions for Asynchronous GPU Tensor Computations”, CC '26, DOI 10.1145/3771775.3786277。
  - **类型。** 类型修饰包括：future（`!T`、`!mbar T`、`!group T`），aref（`%T`：缓冲区加 full/empty mbarrier），迭代器（`#[D].T`）。
  - **对循环的处理。** `#[7].%!<A,B>` 表示 7 个 aref 组成的环，类型里**没有游标**，因此天然循环不变。`iter.next()` 用一个运行时 parity 变量处理回绕；移位算子按相对游标的偏移访问槽。这就是把“轮转坐标系”做成了表层类型。
  - **局限。** 推断规则只在补充材料里，没有可靠性定理；正文也看不出它会静态拒绝两次 `get` 或漏掉的交还。
- **CUTLASS/CuTe 的 `PipelineState`、`PipelineTmaAsync`** [V，官方文档 “Synchronization primitives”]
  - **游标在运行时。** `(index, phase, count)` 是运行时对象：`++` 把 index 对 Stages 取模递增，回绕时翻转 phase；`producer_acquire`/`consumer_wait` 按它等待。
  - **与本设计的关系。** 它是轮转坐标系在运行时的实现，循环只携带一个固定类型的值。没有静态协议检查，两次等待、漏交还都可能发生。
  - 不同版本的编码可能不同（有资料写成 `index + phase*stages`）。
- **Tawa** [V]
  - H. Chen, E. Gaburov, J. Knight, B. Fan, M. Masuda, Z. Zhang, A. Collins, M. Brookhart, V. Grover, B. Hagedorn, C. Sullivan, “Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References”, CGO 2026, arXiv:2510.14719。
  - **做法。** aref 是单槽通道，有 put/get/consumed 的操作语义。深度 D 的环是一组循环使用的 aref，槽号 `k mod D`、parity 都在运行时计算。
  - **局限。** 没有静态类型或顺序检查，正确性依赖生成器本身。
- **Descend** [V]
  - B. Köpcke, S. Gorlatch, M. Steuwer, “Descend: A Safe GPU Systems Programming Language”, PLDI 2024, DOI 10.1145/3656411。
  - **做法。** 用所有权、视图和执行资源类型。`sync` 只有块级，每道屏障都重置借用状态，所以循环不变。
  - **局限。** 没有异步拷贝、mbarrier 或多缓冲。
- **Kuiper** [V]
  - G. Martínez, B. Köpcke, J. Fiala, G. Ebner, T. Ramananandro, M. Steuwer, T. Sorensen, N. Swamy, “Kuiper: Correct and Efficient GPU Programming with Dependent Types and Separation Logic”, PLDI 2026, DOI 10.1145/3808280。
  - **做法。** 基于 Pulse/F\* 的程序逻辑。屏障令牌 `barrier_token(p,q,i,it)` 按**绝对**迭代号 `it` 索引；到达屏障时，线程放出 `p(i,it)`、得到 `q(i,it)`，令牌推进到 `it+1`。可接纳性要求对所有 `it` 都有 `⋆_i p ⊢ ⋆_i q`，取模算术藏在 `p`、`q` 里。
  - **局限。** 没有 mbarrier 或异步流水的模型。
  - **意义。** 这是“绝对下标 + 证明义务”路线的代表，恰好与本报告的相对下标路线相对。
- **Koska 等** [V，确实存在]
  - L. Koska, T. Koska, B. Koska, “Formalizing PTX in Lean to Prove GPU Kernel Correctness”, arXiv:2611.04447, 2026-11-03。
  - **内容。** Lean 4 中的 PTX 小步语义，证明了含两级 TMA/mbarrier 的 Hopper GEMM、FlashAttention 和 MoE 流水。是语义加逐 kernel 的证明，不是类型系统。
  - 公开文本里没有找到 parity 不变量的细节。
- **ThunderKittens** [V 存在，机制 M]
  - B. F. Spector, S. Arora, A. Singhal, D. Y. Fu, C. Ré, “ThunderKittens: Simple, Fast, and Adorable AI Kernels”, ICLR 2025, arXiv:2410.20399。
  - phase 位由人工维护，环游标在运行时，没有类型。
- **其他**（都不给 mbarrier 相位定类型）：
  - Twill（arXiv:2512.18134）[V 存在]；
  - Cypress（arXiv:2504.07004）[V 存在]；
  - Prism（arXiv:2511.11939）[V]：给线程粒度定类型。
  - 定向检索**没有找到**发表过的、给 mbarrier 相位定线性或 typestate 类型的系统。[推测] 这是空白。
- **MimIR 基础论文**（`cgo27-paper248.pdf`，“Principles for Multi-Level Compiler Design”）
  - 不涉及异步流水。
  - 但其中“未能消去的义务以残余 `refly.check` 保留、由流水线报告”的机制，正是“擦除幽灵下标、保留残余验证”的同一模式，可在论文中呼应。

### 5.2 线性与 typestate 语言

- **ATS** [M]：H. Xi, “Applied Type System”, TYPES 2003。
  - 线性 view 按静态整数索引，循环需要显式的静态不变量（存在打包加终止度量）。
  - 环缓冲通常在 `i mod n` 处把数组 view 一分为二，`mod` 的证明放在约束域里。
- **DML** [M]：H. Xi, F. Pfenning, “Eliminating Array Bound Checking Through Dependent Types”, PLDI 1998。循环下标在循环类型中以存在量化出现。
- **Idris 2 / QTT** [M]：E. Brady, “Idris 2: Quantitative Type Theory in Practice”, ECOOP 2021。
  - 重数 0 的下标被擦除，重数 1 的资源保留线性；带推进下标的循环写成对下标的结构递归，即索引 fold。
  - 它是方案 B 最接近的现成对应。
- **Alms** [M]：J. A. Tov, R. Pucella, “Practical Affine Types”, POPL 2011。仿射 typestate 用抽象类型加状态幻影表示，循环要求入口和出口状态相同，正好对应 D′。
- **Mezzo** [M]：F. Pottier, J. Protzenko, “Programming with Permissions in Mezzo”, ICFP 2013。循环写成递归函数，参数的权限即不变量。
- **Linear Haskell** [M]：J.-P. Bernardy, M. Boespflug, R. R. Newton, S. Peyton Jones, A. Spiwack, POPL 2018。
- **Granule** [M]：D. Orchard, V.-B. Liepelt, H. Eades III, “Quantitative program reasoning with graded modal types”, ICFP 2019。适合“恰好 n 次”，不适合模相位。
- **Wadler** [M]：P. Wadler, “Linear types can change the world!”, IFIP TC2 Working Conference, 1990。只有线性更新，没有索引。
- **Rust 嵌入式 DMA** [M，部分 V]：
  - `embedded-dma` 与各 HAL 的 `Transfer` 取走缓冲区所有权，由 `wait()` 归还。
  - stm32f1xx-hal 的 `CircBuffer` 把可读的半区放在运行时字段里，类型循环不变。

### 5.3 验证框架与幽灵状态

- **Meta XROS 的 IPC 队列** [V]：Q. Carbonneaux, N. Zilberstein, C. Klee, P. W. O'Hearn, F. Zappa Nardelli, “Applying Formal Verification to Microkernel IPC at Meta”, CPP 2022, arXiv:2207.06574。
  - Iris/Coq 证明无锁环队列。
  - 关键做法：把模环展开成无限长的逻辑“缎带”，不变量按绝对位置陈述，`mod` 只出现在映射到物理槽的地方。
  - 这是“绝对幽灵下标、物理槽 = n mod D”的范例，对应我们的 Lean 层。
- **Verus** [V 存在]：A. Lattuada, T. Hance, C. Cho, M. Brun, I. Subasinghe, Y. Zhou, J. Howell, B. Parno, C. Hawblitzel, “Verus: Verifying Rust Programs using Linear Ghost Types”, OOPSLA 2023, DOI 10.1145/3586037。
  - 线性幽灵权限在编译时擦除，正是“擦除幽灵令牌”的模型。
  - 论文中是否有环缓冲例子 [U]。
- **Iris / RustBelt** [M]：R. Jung 等, “Iris from the ground up”, JFP 2018；R. Jung, J.-H. Jourdan, R. Krebbers, D. Dreyer, “RustBelt”, POPL 2018。权威/片段与分数幽灵状态是为生产者、消费者各配一个单调计数和逐槽令牌的标准做法。
- **屏障的资源交换**：
  - A. Hobor, C. Gherghina, “Barriers in Concurrent Separation Logic”, ESOP 2011 [M]；
  - S. Blom, M. Huisman, M. Mihelčić, “Specification and verification of GPGPU programs”, SCP 2014, DOI 10.1016/j.scico.2014.03.013 [V]（VerCors）。
- **GPUVerify 的屏障不变量** [V]：N. Chong, A. F. Donaldson, P. H. J. Kelly, J. Ketema, S. Qadeer, “Barrier Invariants: A Shared State Abstraction for the Analysis of Data-Dependent GPU Kernels”, OOPSLA 2013, DOI 10.1145/2509136.2509517。循环需要循环不变量，没有异步屏障。
- **Low\*/F\*** [M]：J. Protzenko 等, “Verified Low-Level Programming Embedded in F\*”, ICFP 2017。
- **其他** [M/U]：Why3 的 VSTTE'12 环形队列 [V] 用逻辑序列加取模物理下标。Prusti、Creusot、CN、Vale、Dafny 都用绝对计数器上的循环不变量，没有找到 GPU 流水实例 [U]。

### 5.4 会话类型与有界缓冲

- **k-MC** [V]：J. Lange, N. Yoshida, “Verifying Asynchronous Interactions via Communicating Session Automata”, CAV 2019, arXiv:1901.09606。
  - k 有界的安全性与穷尽性检查，对应深度 k 的环。
  - 循环是自动机中的环，类型里没有下标。
- **有界异步子类型** [V]：M. Bravetti, M. Carbone, G. Zavattaro, “On the Boundary between Decidability and Undecidability of Asynchronous Session Subtyping”, TCS 2018, arXiv:1703.00659。
- **[M]**：
  - P.-M. Deniélou, N. Yoshida, ICALP 2013 与 CONCUR 2010（缓冲界分析）；
  - S. J. Gay, V. T. Vasconcelos, “Linear type theory for asynchronous session types”, JFP 2010。
  - “Bocchi 的有界缓冲子类型”这一具体论文未核实 [U]。
- **要点。** 会话类型用 μ 递归处理循环，会话类型本身与迭代无关。它之所以可行，正是因为槽号和相位不在类型里，与 D′ 同构。

### 5.5 循环下标状态与软件流水验证

- **Tristan 与 Leroy** [V]：J.-B. Tristan, X. Leroy, “A simple, verified validator for software pipelining”, POPL 2010, pp. 83–92, DOI 10.1145/1706299.1706311。
  - 对每个实例做符号求值，核对“序言 + 稳态核 + 尾声”与按深度展开的原循环一致；验证器本身在 Coq 中证明。
  - 它是“按深度展开检查一圈、保留残余验证”（以及本报告方案 F）的直接先例。
- **Liquid Haskell** [M]：N. Vazou 等, “Refinement Types for Haskell”, ICFP 2014。循环不变量是递归函数参数上的细化，由 SMT 处理取模，非线性 `mod` 很脆弱。这说明方案 C 即便可做也不稳。

### 5.6 谁真正实现了“带索引的线性环状态”，循环怎么处理

- **没有一个 GPU 系统做了。**
  - Async Graphene、Tawa、CUTLASS、ThunderKittens 都把游标和相位放在运行时，类型（如果有）与游标无关。
  - Kuiper 用绝对迭代号，加对所有 `it` 的证明义务，而且只覆盖 `__syncthreads`。
- **通用验证里有两种做法：**
  - Iris（XROS 缎带）和 Verus：用**绝对**幽灵下标，循环靠不变量；
  - Alms、会话类型：类型**与下标无关**，循环靠同一状态进出。
- **对我们最重要的五项：**
  1. Async Graphene：相对坐标的直接先例，也正好是它的空白所在——没有线性、没有可靠性定理。
  2. CUTLASS `PipelineState`：轮转坐标系的运行时实现。
  3. Kuiper（及 Hobor–Gherghina）：绝对下标路线的对照。
  4. XROS 缎带 + Verus：“绝对幽灵、取模只在物理边界”与“幽灵线性擦除”，对应我们的 Lean 层和擦除。
  5. Tristan–Leroy：逐实例验证的正当性。

## 6. 推测与未验证的部分

- §2.3 只在 CPU 后端的小循环上验证了单位分量被消除，GPU 路径没有测。
- 方案 F 的行数和误拒率是估计，没有实现。
- `ws_fold` 依赖 `persistent_fill i ≡ i (mod depth)`。现有 `same_slot` 能否证出它，没有测试：
  - `nat_form` 会把 `i % depth` 当作原子；
  - 但比较的另一侧是 `idx (block % depth)`，形状不同，可能需要小改。
- “没有 GPU 系统给 mbarrier 相位定线性类型”的判断，基于本次检索，不能排除遗漏。
