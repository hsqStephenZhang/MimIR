# 单 CTA 的 TMA / WGMMA 环

本项目给出 `gpu.ring_fold_at` 的单环操作语义和安全证明。一个 CTA 在私有共享内存中用 `d` 个互不重叠的槽处理 `count` 个逻辑块。一个逻辑生产者发起 TMA 填充，一个逻辑消费者按块提交 WGMMA 组。每块的传输字节数和组内 MMA 条数都是任意的自然数函数。

**深度边界影响可完成性。** 安全定理覆盖所有正深度。对源码中的 fold，存在排空执行的充要条件是 `d ≥ 2 ∨ count = 0`。最小反例是 `d = 1, count = 1`：序言发起零次填充，块 0 的等待在第一次补填之前，初始屏障没有事件可使它翻相。任意次重试都停在这里。实际的 `LowerToMem` 用 `depth > 1` 选择流水线。参见 [fold 源码](../../src/mim/plug/gpu/gpu.mim) 和 [lowering 的深度判断](../../src/mim/plug/tensor/phase/lower_to_mem_stage.cpp)。

## 程序与硬件的边界

序言无条件填充 `0 ≤ j < d - 1`。第 `i` 步先等槽 `i % d` 的本代，然后按下表执行。交还的都是上一步的槽，第 0 步的交还为空。补填的块号为 `i + (d - 1)`，只在它小于 `count` 时发起。模型在退出时追加 `wait_group 0`。Mim 只在 `lag = 1` 时发出这条指令。`lag = 0` 的每次提交已紧跟一次 `wait_group 0`。

| 参数 | 等到本块之后的操作顺序 |
| --- | --- |
| `lag = 0` | 交还、补填，再发射本块的 MMA、提交、等待 0 组 |
| `lag = 1` | 发射本块的 MMA、提交、等待至多最新 1 组，再交还、补填 |

`exact` 在源码中约束循环补填。若单独实例化骨架使 `count < d - 1`，序言仍会发起额外的填充。模型保留这些事件，排空见证也使它们完成。块标签仅表示数据来源；在这种实例上读取额外块的全局内存地址是否有效，需另行检查。生成器按请求深度和块数的较小者选择有效深度。参见 [索引与深度定义](../../src/mim/plug/tensor/tensor.mim)。

### 硬件假设

以下假设由 `Device`、`HW` 和 `Host` 的归纳构造子给出。项目自定义的 Lean 公理为 **零**。用 `#print axioms` 检查主要定理，依赖项只有 Lean 标准逻辑中的 `propext`、`Classical.choice` 和 `Quot.sound`。

- 每槽的 full mbarrier 从相位 0 开始，每代需要一次 arrival。一次 fill 记录目标槽、源块、代和预期字节总数。一个块的多条 TMA 指令合并计数，各块之间可任意交错；同块的字节可以分批完成。arrival 和字节到达可按任意顺序发生。两项均齐备才允许发布数据、递增相位并清除在途事务。发布是一次原子事件。[NVIDIA PTX 的 mbarrier 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- 消费者只用奇偶位轮询，未完成时可重试。无界相位计数是证明用的影子状态。设备发布的数据标签取自实际在途的 copy；只有通过等待后，fold 才能发射读取该槽的 MMA。[NVIDIA PTX 的等待指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
- 每条 MMA 登记所读槽和代；commit 把所有待提交的读划入同一组。各组可独立完成，整组完成即归还组内全部读。第 `k` 次提交后（共已提交 `k` 组，编号从 0 起），`wait_group N` 返回要求所有仍在途组的编号至少为 `k - N`。例如最新组已经完成时，一个更老的组仍能阻塞 `wait_group 1`。这条规则保留了“最新”的含义。[NVIDIA PTX 的 WGMMA 提交与等待](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)

## 已证定理

定理在 `Ring` 命名空间内。`Run` 是从初态开始按小步语义执行得到的任意有限前缀。硬件完成可穿插在任意两条指令之间，也可发生在轮询期间。安全性量化所有这些前缀。发射 fill 和 MMA 的规则本身没有“上一代已读完”或“数据正确”的前置条件。

| 定理 | 结论 |
| --- | --- |
| `foldInvariant`、`Certified.step` | 可达状态满足滑动窗口和组编号不变式，或处于上述单槽初始等待。指令、设备完成和重试各步保持已认证的状态。 |
| `no_ABA` | 轮询块 `i` 时，槽的相位只可能是 `g = i / d` 或 `g + 1`。奇偶测试成功当且仅当块 `i` 已发布且没有在途写。因此两次翻相不会冒充本代完成。 |
| `read_current_generation`（T1） | 发射任何一条读块 `i` 的 MMA 之前，槽 `i % d` 中可见的数据正是块 `i`、代 `i / d` 的填充，且相位已翻转。 |
| `overwrite_after_last_read`（T2） | 发射块 `j ≥ d` 的填充之前，块 `j - d` 的读已经全部完成。待提交列表和任何在途组里都没有这一读。首轮填充没有上一代。 |
| `release_after_last_read` | 每次交还一个实际读过的槽时，该块的所有读已经完成。 |
| `exists_drained_iff`、`singleSlot_stalls` | 对正深度，恰在 `d ≥ 2` 或空循环时存在排空执行。见证处理了全部 `count` 块，指令已用尽，待提交列表、在途组及每槽事务均为空。单槽非空循环的每个可达状态都等于初态。 |

不变式用区间 `[i, i + d)` 给各物理槽指定唯一的逻辑块。已等待的块必须可见，已发射但尚未等待的块可处于传输中或已发布，其余槽准备接收该代。窗口推进一格时，退出的块与新进入的块恰好相差一圈。另一部分不变式把每个在途组的读关联到提交该组的迭代，并记录已由等待指令保证完成的组编号下界。存在性证明选择一种及时完成传输和组的调度；安全证明允许任意延迟。

## 构建与复核

在本目录执行以下命令。项目固定 Lean 4.36.0，使用 Lake 5 和标准库，无外部依赖。

```bash
export PATH="$HOME/.elan/bin:$PATH"
lake build
lake env lean --stdin <<'LEAN'
import Ring
#print axioms Ring.read_current_generation
#print axioms Ring.overwrite_after_last_read
#print axioms Ring.no_ABA
#print axioms Ring.exists_drained_iff
LEAN
```

硬件事件和 CTA 的指令小步在 [Machine.lean](Ring/Machine.lean)，窗口与保持性在 [Invariant.lean](Ring/Invariant.lean)，指令序列证书及其语义在 [Certificate.lean](Ring/Certificate.lean)。[Fold.lean](Ring/Fold.lean) 构造程序并证明它具有证书，[Ring.lean](Ring.lean) 给出对任意执行的定理和单槽反例。

## 与残余 fold 对接

[lit 用例](../../lit/gpu/ring_fold_trace.mim)用与 lowering 相同的 `tma_ring`、`pipeline_plan.full_parity`、`refill_block` 和 `ring_fold` 实例化单 CTA 环。每个物理槽有一个独立的面板标签，普通循环须正确轮转这些标签，展开循环须选对字面槽。每次 fill 用 `tma_fill` 复制一行 128 字节，每步用 `wgmma_batch` 发射两条 MMA。MMA 描述符在用例中携带块号和面板偏移，便于直接核对读取来源。

先运行 `gpu.check_sync`，再用 Mim 重写规则把特化后的叶操作换成消费同一条内存链的事件记录。这样能在擦除前观察 `retire` 的实际槽和 live 位。将残余程序唯一的记录原语绑定到主机打印函数后，运行其循环、索引和分支。提取器无需重写一遍调度，也不依赖 dump 的节点号或定义的打印顺序。[驱动脚本](../../lit/gpu/ring_fold_trace.py)

| 模型 | 核对的残余事件及操作数 |
| --- | --- |
| 初态 | leader 对每个 full barrier 初始化一次，arrival 数为 1。面板与屏障分别从各自数组的偏移 0 编号。 |
| `fill j`（含补填） | `mbar_expect_tx_if` 的屏障偏移、128 字节，紧接着 `tma_load_if` 的源行 `j`、目标面板偏移 `128 * (j % d)` 和同一屏障偏移 `8 * (j % d)`。 |
| `poll i` | `mbar_wait_at` 使用槽 `i % d` 的屏障和奇偶位 `(i / d) % 2`。记录一次程序等待，硬件重试属于小步语义。 |
| `mma i` / `commit` / `waitGroups n` | 逐条核对 `wgmma_f16` 的块与物理面板标签、`wgmma_commit` 和 `wgmma_wait` 的立即数。 |
| `release none` / `release (some j)` | `retire` 的 `(live, slot)` 分别为 `(0, 0)` 和 `(1, j % d)`。 |
| `next` | 循环推进，无独立的设备同步事件。CTA fence、WGMMA fence 和收尾的 `async_wait` 均在这次投影之外。 |

从本目录运行 `lake exe ring_trace 3 7 1 2` 可打印相同格式的 Lean 参考序列，参数依次为深度、块数、lag 和每块的 MMA 条数。输出直接遍历 `Ring.fold`。[可执行投影](Trace.lean)

**56 个实例。** 深度取 2、3、4，块数取去重后的 `{0, 1, d−1, d, 2d+1}`，lag 取 0、1，turns 取两种值，width 固定为 2。每对 rolled / unrolled 的观察序列相同。`lag = 1` 的 28 个实例与 Lean 逐条相等。`lag = 0` 的 28 个实例都恰好少了模型最后的 `wait_group 0`。非空循环的最后一组已在步内等待，空循环从未提交过组。测试只接受精确相等或这一项已知差异。两边都保留 `count < d−1` 时序言超出逻辑块数的填充。

这是对具体特化实例的同步序列验证。把安全定理直接用于物理 kernel 仍需证明描述符、实际读写地址与字节范围、线程收敛以及擦除到硬件小步的对应。末尾冗余等待的语义等价也尚未在 Lean 中证明。

### 生成器的覆盖范围

在仓库根目录运行以下命令。分类器按一次 lowering 分配的环计数。对于协议形状符合的环，再以同一次编译的 panel group 查 NVPTX 的实际槽区间，缺少布局证据时返回 `unknown-layout`。

```bash
(ulimit -v 16000000; bench/tools/compile_all.sh /tmp/ring-bench -X tensor:ring-coverage=1)
python3 bench/tools/ring_coverage.py /tmp/ring-bench
```

| 73 个成功编译的 benchmark 中的主分类 | 环数 |
| --- | ---: |
| 满足单 CTA、单生产者、单消费者、TMA / WGMMA、正深度、允许的 lag、区域不叠放 | 0 |
| `cp.async`，使用复制组同步 | 5 |
| 普通 TMA，两个消费 warpgroup | 2 |
| cluster（含 split） | 2 |
| 独立生产者的 `ws_fold` | 1 |
| persistent tiles | 1 |
| **合计** | **11** |

11 个环各在一个 benchmark 中，其余 62 个没有走这两处造环路径。另有 10 个拒绝用例按预期失败。表中每个环只计一次。两种普通 TMA 环中，F16 输出的 `gemm_tiled_wg_tma_h` 还与尾声发生物理叠放。把这两个 GEMM 的 CTA 从两组缩为一组后，F32 输出的环满足前提，F16 输出的环仍因叠放而排除。[覆盖回归](../../lit/gpu/ring_fold_coverage.mim)

多消费者需要分别跟踪各组的在途读和最后一次交还。`ws_fold` 还引入独立生产者与 empty barrier。cluster 需要跨 CTA 跟踪 multicast 和远端 arrival，persistent 需要跨 tile 保持代号、奇偶位和初始化状态。叠放区域需要证明相邻使用者的生存期及同步边界。[设计文档 §14.3、§17](../../docs/linear-types-design.zh-CN.md)
