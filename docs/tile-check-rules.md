# tile 层级规则：用 Mim 表达、由 PE 判定（2026-11-13）

## 机制

`tensor.mim` 的 `mod check`（提交 f34347640）把 tile 请求中依赖根运算的合法性规则写成 Mim `lam`，由部分求值（PE）在 elaboration 时求出：

- C++ 只做反射：`check.parallel` / `check.inputs` / `check.extents` 读出根的并行循环数、输入个数、各循环长度；运行期长度读成 0，表示未知。
- 规则沿层级链 `match` 递归，返回第一条违反的规则编号；报错文字在 `check.messages`。
- `tile_req` 的 normalizer 只构造 `check.levels (op, levels)`，结果是非零字面量就在调用处报错。
- 递归函数带 filter `@(core.pe.is_closed l)`：链是闭项才展开，定义体里和符号链上保持未归约。
- 块大小为 0（未知）时规则放过，一条规则判定不了不会连带其它规则；放过的部分由 `TileFuse` / `lower_to_mem` 晚检兜底。

已迁移 14 条规则（每条有 `lit/error/` 测试）：并行层不切归约循环、归约层只切一个循环、promote 位、lane 数、MMA 整除块、MMA 前提升 A/B、
cluster 整除 work-group 数、split 块数、specialized 不在 cluster 下、persistent 下 TMA 须 specialized、persistent ring 整除块数、bound 层不 peel 尾块、向量层尺寸。

## 编译时间代价

迁移前（0983deae3）与迁移后（f34347640）两份 release 构建交替运行，各 15 轮取中位数：

| 场景 | 迁移前 | 迁移后 | 差值 |
|---|---:|---:|---:|
| 只加载 tensor 插件（空程序） | 163.4 ms | 171.4 ms | +8.0 |
| elaboration：linear_gelu pws | 180.1 | 192.6 | +12.4 |
| elaboration：addmm tma | 178.2 | 190.1 | +11.8 |
| elaboration：softmax | 185.4 | 198.1 | +12.7 |
| elaboration：cross_entropy | 194.5 | 209.4 | +14.9 |
| 完整编译：linear_gelu pws | 1109.1 | 1125.8 | +16.8 |
| 完整编译：addmm tma | 371.8 | 387.4 | +15.6 |
| 完整编译：softmax | 677.9 | 688.2 | +10.3 |
| 完整编译：cross_entropy | 699.2 | 711.5 | +12.3 |

拆分：约 8 ms 是加载变大的 `tensor.mim`（所有加载 tensor 插件的程序都付），约 4–7 ms 是 elaboration 时的 PE 检查，约 0–5 ms 是后续 phase 重建 `tile_req` 时再次检查。
后一部分可以靠只在 elaboration 时检查、或按 (op, levels) 缓存去掉；目前占编译时间 1.5–4%，未优化。

## 表达能力

### 计算上够用

PE 下的 Mim 有一般递归（`go` + filter，嵌套深度受 `--pe-fuel` 限制，默认 1024，层级链通常不到 10 层）、Nat 算术、`vec.fold`、对 nom 的 `match` 和依赖类型。
对字面量的层级值和反射出来的数字，任何可判定谓词都能写。

### Mim 拿不到信息的检查

| 类别 | 例子（现有 C++ 检查） | 原因 |
|---|---|---|
| a. 根的项结构 | 组合子是不是 `add`/`max`，`proj_map` 是否线性，root 把循环 d 写到输出轴 d（`canonical_write`） | Mim 不能解构 Def；`refly` 只有不透明的 `Code`、`type`、`gid` 和断言式的 `alpha.e`，没有按节点种类和操作数匹配的能力 |
| b. 语义 / 等式性质 | 组合子可重结合、种子是单位元 | 一般不可判定；C++ 也只是按语法识别已知规约律，再用 α 等价比较单位元 |
| c. 后续 phase 的决定 | A 须 K-major、TMA 行 128 字节（依赖 promotion 选的 layout 方向）、multicast 行数份额、smem / 寄存器预算 | elaboration 时这些信息还不存在 |
| d. 运行期值 | 符号尺寸 | 静态不可判定 |
| e. 目标硬件 | sm 版本、smem 容量 | 目前只在 C++ 的 flags 里 |

### 机制本身的语义缺口

- 算不出字面量结果就放过：链不是闭项，或用到的尺寸只在运行期才知道时，当作合法；健全性靠保留的 C++ 晚检。
- 两份规则的一致性没有证明：Mim 规则相当于规格，`lower_to_mem` 是实现。
  Mim 放过而 lowering 拒绝会被晚检兜住；Mim 拒绝而 lowering 能处理则是误报。目前只靠 lit 约束。
- PE 的限制：
  - 带隐式参数的 lam 在自身体内递归，或在 `mutual` 中被前向引用时，调用曾被归约到一个空壳 lam 上；
    已在 `World::app` 修复：外层 lam 的体是尚未 set 的内层 lam 时不展开（`lit/pe_implicit_rec.mim`），规则不再需要体内的 `go` 包装；
  - 元组选择 `(a, b)#c` 两边都求值，递归只能放在 `match` arm 里；
  - 递归规则要带 `@(core.pe.is_closed l)` filter，否则 arm 引用了 lam 自身的变量时，每次 PE 都重建 arm，在符号链上无限展开（与 `lit/docs/iter.mim` 的写法一致）；
  - normalizer 分不清规则自己的变量和用户的符号尺寸，所以未知长度在反射处记为 0；
  - 规则返回 `Verdict = «3; Nat»`（规则编号和两个值），`messages` 用 `{}` 占位，报错带上具体数值，例如 “2 work-groups do not form clusters of 4”。
- 同一 `mod` 的 annex 共用一个 tag；插件最多 256 个 tag。

## 处理方式（分层）

| 层 | 适用 | 失败时 |
|---|---|---|
| 类型 | 层级结构、操作数种类、tile 形状一致 | elaboration 时类型错误 |
| Mim 规则 + PE | 字面量的层级值与数值约束 | elaboration 时在调用处报错 |
| C++ 识别 + 反射公理，规则仍在 Mim | a、b、e：C++ 只给数据（规约律编号、单位元、sm 版本），判定写在 Mim | 识别不出返回“未知”，规则放过 |
| phase 中的 C++ 晚检 | c，以及以上各层放过的情况 | 默认警告并回退（相当于 CUTLASS `can_implement`），`-X tensor:strict=1` 下报错 |
| 运行期 | d | 暂时假定有 runtime assert |

第三层的现成先例是 `root_contraction`：C++ 识别收缩模式，返回一个类型给 Mim。

## 下一步

1. 组合子检查迁到 Mim：加反射公理 `check.law op` 返回识别出的规约律编号（识别不出为 0），可重结合与种子是单位元的判定写进 Mim 规则；
   单位元比较需要 Bool 版的 α 等价，`refly.alpha` 目前只是断言，可顺带加一个。
2. c 类前移：先把 promotion 的 layout 选择改成 Mim 函数，“A 须 K-major”“TMA 128 字节”就成为对它结果的检查，可在 elaboration 时完成；改动较大。
3. 消除两份规则：让 `lower_to_mem` 的晚检直接调用同一个 `check.levels`，符号情况到 lowering 时已经是字面量。
