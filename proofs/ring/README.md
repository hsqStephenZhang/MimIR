# 单 CTA 的异步共享内存环

73 个 GPU benchmark 生成 11 个环。按单 CTA、槽区间不叠放等前提，**7 个环原则上可应用协议定理**，包括 5 个 `cp.async`、1 个双消费者合作 TMA 和 1 个独立生产者 `ws_fold`。分类器现按模型报告这 7 个环（`covered-async 5`、`covered-cooperative 1`、`covered-warp_specialized 1`）。把协议结论用于物理 kernel 时，还要核对读写地址、线程参与范围与生成指令。下文给出各路径的假设和分类口径。

## 单消费者 TMA 小步模型

一个 CTA 在私有共享内存中用 `d` 个互不重叠的槽处理 `count` 个逻辑块。一个逻辑生产者发起 TMA 填充，一个逻辑消费者按块提交 WGMMA 组。每块传输字节数和组内 MMA 条数都是任意的自然数函数。

**深度边界影响可完成性。** 安全定理覆盖所有正深度。对源码中的 fold，存在排空执行的充要条件是 `d ≥ 2 ∨ count = 0`。最小反例是 `d = 1, count = 1`：序言发起零次填充，块 0 的等待在第一次补填之前，初始屏障没有事件可使它翻相。任意次重试都停在这里。实际的 `LowerToMem` 用 `depth > 1` 选择流水线。参见 [fold 源码](../../src/mim/plug/gpu/gpu.mim) 和 [lowering 的深度判断](../../src/mim/plug/tensor/phase/lower_to_mem_stage.cpp)。

### 程序与硬件的边界

序言无条件填充 `0 ≤ j < d - 1`。第 `i` 步先等槽 `i % d` 的本代，然后按下表执行。交还的都是上一步的槽，第 0 步的交还为空。补填的块号为 `i + (d - 1)`，只在它小于 `count` 时发起。模型在退出时追加 `wait_group 0`。Mim 只在 `lag = 1` 时发出这条指令。`lag = 0` 的每次提交已紧跟一次 `wait_group 0`。

| 参数 | 等到本块之后的操作顺序 |
| --- | --- |
| `lag = 0` | CTA 会合、交还、补填，再发射本块的 MMA、提交、等待 0 组 |
| `lag = 1` | 发射本块的 MMA、提交、等待至多最新 1 组，再由 CTA 会合、交还、补填 |

`exact` 在源码中约束循环补填。若单独实例化骨架使 `count < d - 1`，序言仍会发起额外的填充。`prologue_extra_fill` 证明每个 `count ≤ j < d−1` 的块都出现在序言。循环中的 `pipeline_plan.refill_block` 把源块钳到 `count−1`，TMA 的 `exact = true` 使越过末块的循环补填完全跳过；序言直接传 `n`，不经过该钳位。模型保留这些事件，排空见证也使它们完成。块标签仅表示数据来源；在这种实例上读取额外块的全局内存地址是否有效，需另行检查。生成器按请求深度和块数的较小者选择有效深度。参见 [索引与深度定义](../../src/mim/plug/tensor/tensor.mim)。

### 硬件假设

以下假设由 `Device`、`HW` 和 `Host` 的归纳构造子给出。项目自定义的 Lean 公理为 **零**。用 `#print axioms` 检查主要定理，依赖项只有 Lean 标准逻辑中的 `propext`、`Classical.choice` 和 `Quot.sound`。

- 每槽的 full mbarrier 从相位 0 开始，每代需要一次 arrival。一次 fill 记录目标槽、源块、代和预期字节总数。开始写入时原有 `visible` 立即失效。一个块的多条 TMA 指令合并计数，各块之间可任意交错；同块的字节可以分批完成。arrival 和字节到达可按任意顺序发生。两项均齐备才允许发布数据、递增相位并清除在途事务。发布是一次原子事件。[NVIDIA PTX 的 mbarrier 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- 消费者只用奇偶位轮询，未完成时可重试。无界相位计数是证明用的影子状态。设备发布的数据标签取自实际在途的 copy；只有通过等待后，fold 才能发射读取该槽的 MMA。[NVIDIA PTX 的等待指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
- 每条 MMA 登记所读槽和代；commit 把所有待提交的读划入同一组。各组可独立完成，整组完成即归还组内全部读。第 `k` 次提交后（共已提交 `k` 组，编号从 0 起），`wait_group N` 返回要求所有仍在途组的编号至少为 `k - N`。例如最新组已经完成时，一个更老的组仍能阻塞 `wait_group 1`。这条规则保留了“最新”的含义。[NVIDIA PTX 的 WGMMA 提交与等待](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)

### 单消费者定理

定理在 `Ring` 命名空间内。`Run` 是从初态开始按小步语义执行得到的任意有限前缀。硬件完成可穿插在任意两条指令之间，也可发生在轮询期间。安全性量化所有这些前缀。发射 fill 和 MMA 的规则本身没有“上一代已读完”或“数据正确”的前置条件。

| 定理 | 结论 |
| --- | --- |
| `foldInvariant`、`Certified.step` | 可达状态满足滑动窗口和组编号不变式，或处于上述单槽初始等待。指令、设备完成和重试各步保持已认证的状态。 |
| `no_ABA` | 轮询块 `i` 时，槽的相位只可能是 `g = i / d` 或 `g + 1`。奇偶测试成功当且仅当块 `i` 已发布且没有在途写。因此两次翻相不会冒充本代完成。 |
| `read_current_generation`（T1） | 发射任何一条读块 `i` 的 MMA 之前，槽 `i % d` 中可见的数据正是块 `i`、代 `i / d` 的填充，且相位已翻转。 |
| `overwrite_after_last_read`（T2） | 发射块 `j ≥ d` 的填充之前，块 `j - d` 的读已经全部完成。待提交列表和任何在途组里都没有这一读。首轮填充没有上一代。 |
| `live_read_race_free` | 在任意可达状态，每个待提交或在途组内的读 `r` 所指定的物理槽都满足 `visible = some r ∧ tx = none`，包括指令之间和轮询重试期间。 |
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
#print axioms Ring.live_read_race_free
#print axioms Ring.wrong_panel_rejected
#print axioms Ring.no_ABA
#print axioms Ring.exists_drained_iff
#print axioms Ring.Async.read_current_generation
#print axioms Ring.Async.overwrite_after_last_read
#print axioms Ring.Async.fold_of_accepted
#print axioms Ring.Async.readCode_of_accepted
#print axioms Ring.Async.accepted_read_current_generation
#print axioms Ring.Async.accepted_overwrite_after_last_read
#print axioms Ring.Async.expected_accepted
#print axioms Ring.Async.diagnose_none
#print axioms Ring.Cooperative.poll_published
#print axioms Ring.Cooperative.read_current_generation
#print axioms Ring.Cooperative.overwrite_after_last_read
#print axioms Ring.EmptyBarrier.handshake
#print axioms Ring.WarpSpecialized.read_current_generation
#print axioms Ring.WarpSpecialized.overwrite_after_last_read
LEAN
```

`fill j s`、`poll i s`、`mma i s` 的机器语义直接使用物理操作数 `s`。只有 `Rule` 的相应构造子把它与逻辑块取模相联系。`wrong_panel_rejected` 证明等待正确、随后 MMA 面板槽错误的指令流无法获得证书。单逻辑消费者自己执行的组等待已足以保证它的 T2；`refilled` 中的 `fence` 在单参与者投影中是一步空操作。骨架中每个单消费者批次显式经过 `wgmmaFence`；模型投影掉寄存器及累加器的访问，只证明共享面板的读写。此项修复本身新增覆盖 0 个 benchmark 环。

硬件事件和 CTA 的指令小步在 [Machine.lean](Ring/Machine.lean)，窗口与保持性在 [Invariant.lean](Ring/Invariant.lean)，指令序列证书及其语义在 [Certificate.lean](Ring/Certificate.lean)。[Fold.lean](Ring/Fold.lean) 构造程序并证明它具有证书，[Ring.lean](Ring.lean) 给出对任意执行的定理和单槽反例。

## 与残余 fold 对接

[lit 用例](../../lit/gpu/ring_fold_trace.mim)用与 lowering 相同的 `tma_ring`、`pipeline_plan.full_parity`、`refill_block` 和 `ring_fold` 实例化单 CTA 环。每个物理槽有一个独立的面板标签，普通循环须正确轮转这些标签，展开循环须选对字面槽。每次 fill 用 `tma_fill` 复制一行 128 字节，每步用 `wgmma_batch` 发射两条 MMA。MMA 描述符在用例中携带块号和面板偏移，便于直接核对读取来源。

先运行 `gpu.check_sync`，再用 Mim 重写规则把特化后的叶操作换成消费同一条内存链的事件记录。这样能在擦除前观察 `retire` 的实际槽和 live 位。将残余程序唯一的记录原语绑定到主机打印函数后，运行其循环、索引和分支。提取器无需重写一遍调度，也不依赖 dump 的节点号或定义的打印顺序。[驱动脚本](../../lit/gpu/ring_fold_trace.py)

| 模型 | 核对的残余事件及操作数 |
| --- | --- |
| 初态 | leader 对每个 full barrier 初始化一次，arrival 数为 1。面板与屏障分别从各自数组的偏移 0 编号。 |
| `fill j s`（含补填） | `mbar_expect_tx_if` 的屏障偏移、128 字节，紧接着 `tma_load_if` 的源行 `j`、目标面板偏移 `128 * s` 和同一屏障偏移 `8 * s`。 |
| `poll i s` | `mbar_wait_at` 使用槽 `s` 的屏障和奇偶位 `(i / d) % 2`。记录一次程序等待，硬件重试属于小步语义。 |
| `mma i s` / `commit` / `waitGroups n` | 逐条核对 `wgmma_f16` 的块与物理面板标签、`wgmma_commit` 和 `wgmma_wait` 的立即数。 |
| `release none` / `release (some j)` | `retire` 的 `(live, slot)` 分别为 `(0, 0)` 和 `(1, j % d)`。 |
| `fence`、`wgmmaFence`、`next` | 前两者在参考程序中显式出现，现有 trace 不打印这两条 fence。`next` 推进循环，收尾的 `async_wait` 也在 trace 投影之外。 |

从本目录运行 `lake exe ring_trace 3 7 1 2` 可打印相同格式的 Lean 参考序列，参数依次为深度、块数、lag 和每块的 MMA 条数。输出直接遍历 `Ring.fold`。[可执行投影](Trace.lean)

**56 个实例。** 深度取 2、3、4，块数取去重后的 `{0, 1, d−1, d, 2d+1}`，lag 取 0、1，turns 取两种值，width 固定为 2。每对 rolled / unrolled 的观察序列相同。`lag = 1` 的 28 个实例与 Lean 逐条相等。`lag = 0` 的 28 个实例都恰好少了模型最后的 `wait_group 0`。非空循环的最后一组已在步内等待，空循环从未提交过组。测试只接受精确相等或这一项已知差异。两边都保留 `count < d−1` 时序言超出逻辑块数的填充。

这是对具体特化实例的同步序列验证。把安全定理直接用于物理 kernel 仍需证明描述符、实际读写地址与字节范围、线程收敛以及擦除到硬件小步的对应。末尾冗余等待的语义等价也尚未在 Lean 中证明。

## 生成器的覆盖范围

在仓库根目录运行以下命令。分类器按一次 lowering 分配的环计数。对于协议形状符合的环，再以同一次编译的 panel group 查 NVPTX 的实际槽区间，缺少布局证据时返回 `unknown-layout`。

```bash
(ulimit -v 16000000; bench/tools/compile_all.sh /tmp/ring-bench -X tensor:ring-coverage=1)
python3 bench/tools/ring_coverage.py /tmp/ring-bench
```

| 原分类器的日志 | 环数 | 原则覆盖 | 依据及限制 |
| --- | ---: | ---: | --- |
| `cp.async` | 5 | 5 | 均为双槽 warp fold；应用到物理 kernel 需核对每线程写入范围及实际布局。 |
| `multiple-consumers` | 2 | 1 | F32 环布局为 `disjoint`，F16 环为 `overlaid`。 |
| `ws_fold` | 1 | 1 | 深度 4，两个消费组，非 cluster、非 persistent，布局为 `disjoint`。 |
| `cluster` | 2 | 0 | 含 split，需要跨 CTA 跟踪 multicast 和远端 arrival。 |
| `persistent` | 1 | 0 | 需要跨 tile 保持代号、相位、释放和初始化状态。 |
| **合计** | **11** | **7** | 其余 62 个 benchmark 没有走这两处造环路径。 |

日志现在给出 `model=` 与 `groups=`。`model` 取 `tma`、`async`、`cooperative`、`warp_specialized` 或 `none`；`none` 时 `scope` 写明原因（`persistent`、`cluster`、`depth`、`ws-config`、`no-wgmma-batch`、`consumers`）。`groups` 列出环各槽的 panel group；不在任何 group 中的槽记为 `shared`，NVPTX 逐个顺序放置它们，因此不会叠放。5 个 `cp.async` 环的四个面板都属此类。合作路径另要求 `consumers=2`、`lag=1`、`count ≥ d−1`；独立生产者要求两个消费组、非 pingpong、`lag=1`。

统计使用 `f8385cf39` 的 LowerToMem 和 NVPTX 日志，并重新运行了 `ring_coverage.py`。日志中的主分类互斥；例如 `multiple-consumers` 的第二条还带有叠放布局。相邻使用者的生存期和同步边界需要另证。[分类脚本](../../bench/tools/ring_coverage.py) · [覆盖回归](../../lit/gpu/ring_fold_coverage.mim)

### 并发协议的证明口径

三个扩展使用事件时间表描述有限执行前缀。`a ≺ b` 表示如果 `b` 发生，`a` 必定已经发生；`none` 表示尚未发生。各自的 `Fold` 结构规定生成骨架中的指令顺序、操作数及硬件等待条件，`ReadCode` 把单次实际读取关联到本地批次或同步 load。T1/T2 量化所有满足这些条件的时间表，允许各参与者和设备交错、停顿。三条路径共用 `Writes.Shows`，它要求目标槽中最新已发起的写具有正确票号和源块，且该槽上所有已发起的写均已完成。`Async`、`Cooperative`、`WarpSpecialized` 各自的 `exists_execution` 另构造了执行到所有块的时间表。[时间表与可见性定义](Ring/Timeline.lean) · [复制组及合作 TMA 见证](Ring/Witness.lean) · [独立生产者见证](Ring/WarpWitness.lean)

现有 56 项残余 trace 比较只针对单消费者 TMA。扩展路径的可信边界包括从实际读写地址到槽和字节片段的映射、warpgroup 的收敛、屏障作用域、初始化可见性，以及实际指令满足相应 `Fold` 和 `ReadCode` 的证据。尚需从新路径的残余指令自动提取这些证据。

### 每线程复制组（第二项，新增 5 个协议实例）

`Async.Fold` 给每个发出复制的线程独立的 issue、commit、complete、wait 和 settle 事件。线程可在会合之间错开，同一线程的组也可乱序完成。每次 fill 各线程各提交一组（包括空组）。第 `i` 次等待前已提交 `d−1+i` 组，`pending=d−2` 保证票号 `0..i` 的组都已完成。各线程随后按需执行 proxy fence，所有线程到达 `refilled` 的 CTA 屏障才放行。末尾 `async_wait` 等待本线程的剩余复制。此路径使用 `exact=false`，因此最后几次补填仍递增票号，源块重复取 `count−1`。

- `Async.read_current_generation` 对每个实际发出复制的线程分别证明：读发射时及读在途期间，其在目标槽负责的部分显示本块、本次票号，且该部分没有在途写。前提是各线程覆盖所读面板、写入区间互不冲突，读与填充操作数通过槽位及源块证书。
- `Async.overwrite_after_last_read` 证明任一线程重写同槽之前，每个较老的同步 load 都已执行，属于 WGMMA 的读所在批次也已完成。`lag=0`，每个相关 WGMMA 批次在此步内等待 0，异步代理读还要求 proxy fence 和 WGMMA fence。CTA 会合把每个读者的本地顺序传到每个填充线程。
- 五条 `fold=warp depth=2 lag=0` 日志分别为 `gpu_attn_attn_{law,roll,warp,wg}` 与 `gpu_fusion_attn_warps`。`wg` 的 PTX 含 proxy fence 和步内 WGMMA 等待，其余四条用同步读。warp lowering 要求块数至少为 2，另行为第二个槽分配共享缓冲区。现有分类脚本对这些环打印 `layout=n/a`。对应 PTX 中，`attn_wg` 的四个静态面板各占 8192 字节，`attn_warps` 的各占 8704 字节；`law/roll/warp` 在动态共享区的四个面板基址偏移分别为 0、17408、34816、52224。这五项是协议层的覆盖，分类器宜记录 warp 环的缓冲区身份和实际区间，核对双槽、复制等待立即数、屏障、代理及读退休方式后计入物理布局已确认的覆盖。仅看 `scope=cp.async` 不足以纳入其他 lowering 路径。

#### 可执行检查器

[AsyncCode.lean](Ring/AsyncCode.lean) 为单个线程按程序顺序看到的指令定义 `Instr`：`fill j s src`（票号 `j` 的一条 `cp.async`，可连续多条）、`commit`、`waitGroup n`、`proxyFence`、`barrier`（`refilled` 的 `bar.sync`）、`read b s r`（`r` 为 `load` 或 `wgmma c`）、`wgmmaFence c`、`wgmmaCommit c`、`wgmmaWait0 c`、`settle`（末尾 `cp.async.wait_all`）。`accepts d count proxy P` 是一次线性扫描。每条指令只对照之前各类指令的计数及前一条指令检查局部规则：票号等于已提交组数，槽为 `j % d`，源块为 `Async.source`；序言恰好 `d−1` 组；第 `i` 步依次为立即数 `d−2` 的等待、（`proxy` 时）proxy fence、屏障、补填 `i+d−1` 及其提交，其后才读块 `i` 的槽 `i % d`；WGMMA 读必须位于本步内编号连续的 fence/commit/wait 0 批次中，且只在 `proxy` 时允许；最后恰好一个 `settle`；`d ≥ 2`。

`expected d count proxy loads mmas` 生成规范序列（每步 `loads` 个同步读，`proxy` 时再加一个含 `mmas` 个读的 WGMMA 批次），`expected_accepted` 证明它被接受。`wrong_slot_rejected`、`wrong_immediate_rejected`、`refill_without_barrier_rejected` 是一般的拒绝引理，文件末尾另有 `decide` 例子。

**时间表。** `Exec` 给每个线程一个已执行前缀长度 `done l` 和每个位置的时刻 `time l q`，另给屏障放行 `release k`、复制组完成 `complete l j`、WGMMA 批次完成 `batchDone l b`。`Consistent prog e` 只要求：

- 程序顺序：已执行前缀内的时刻严格递增；
- 屏障：每个线程的第 `k` 个 `barrier` 早于 `release k`（某线程没有第 `k` 个屏障则永不放行），线程在其后的下一条指令晚于 `release k`；
- 硬件：组完成晚于其提交；位置 `q` 的 `waitGroup n` 按它自己的立即数满足 `Stream.WaitGroup`；`settle` 之前已提交的组均在它之前完成；批次完成晚于其 `wgmmaCommit`，`wgmmaWait0` 之前已提交的批次均在它之前完成。

`cta prog e` 由程序和时间表直接构造 `Async.CTA`：每个线程票号 `j` 的 issue 取该票号**第一条** `fill`，commit、wait、proxy fence、settle 取相应种类的第 `n` 次出现；槽和源块取第一条 `fill` 的操作数；`fence` 即 `release`。warpgroup 投影到线程本身，即每个线程自己的 WGMMA fence/commit/wait 事件代表其所在批次。`readAt prog e l q` 把位置 `q` 的 `read` 变成 `Async.Read`。

- `fold_of_accepted`：各线程程序都被接受且时间表满足 `Consistent`，则 `Async.Fold (cta prog e) d count (d−2) proxy` 成立。
- `readCode_of_accepted`：此时每个 `readAt` 给出的读满足 `Async.ReadCode`。
- `accepted_read_current_generation`、`accepted_overwrite_after_last_read`：由上两条直接得到 T1、T2，对任意被接受的程序和任意一致的时间表成立。`Fold` 的每个字段都由程序顺序、屏障或硬件规则推出，没有削弱。
- `diagnose_none`：可执行文件报告 `accepted` 时，`accepts` 为真。

检查器只核对指令种类、顺序和逻辑操作数。从 PTX 地址到槽号和源块、各线程写入区间互不冲突、所有线程执行同一 `bar.sync`，仍由提取器负责。

**可执行文件。** `lake exe ring_async_check D COUNT PROXY < ops` 读取一个线程的指令，`PROXY` 取 `0/1`。每行一条，`#` 之后为注释，空行忽略：

```text
fill J SLOT SRC | commit | wait N | proxy_fence | bar | settle
read BLOCK SLOT load | read BLOCK SLOT wgmma C
wgmma_fence C | wgmma_commit C | wgmma_wait0 C
```

输出 `accepted`（退出码 0），或 `rejected op I (line L) 'OP': 原因`，序列在末尾才失败时为 `rejected at end (op I): 原因`（退出码 1）；参数或语法错误退出码 2。`lake exe ring_async_check --expected D COUNT PROXY LOADS MMAS` 以同一格式打印规范序列。

### 合作 TMA（第三项，新增 1 个布局不叠放的环）

`Cooperative.Fold` 对应 `consumers=0, cluster=0, persistent=false` 的 `tma_ring` 和 `lag=1, exact=true` 的 `ring_fold_at`，要求 `2 ≤ d` 且 `d−1 ≤ count`。日志的 `consumers=2` 计 CTA 内的读者组；传给 `tma_ring` 的 0 表示这些读者自己填环。leader 发起 TMA，各组独立轮询同一 full barrier，然后各自 fence、发射若干 MMA、提交一组、等待最近 1 组。任意组可先到达 `refilled`。各组都到达后 CTA 屏障才放行，leader 才能向旧槽写入。该路径没有 empty barrier。

- `Cooperative.poll_published` 从实际屏障上的发布次数、成功 poll 的单比特比较和迭代顺序推出相应填充已发布。一次发布要求一次 arrival 和本次预期的全部字节均已到达；时间表将最后一个字节到达之前的细节投影掉。
- `Cooperative.read_current_generation` 证明任意组的任一 MMA 发射时及在途期间读到当前块和代，物理槽上没有在途写。`Cooperative.overwrite_after_last_read` 证明同槽的后续写严格晚于**每个**读者组的旧批次退休。屏障自身不使 WGMMA 退休。证明使用每组每块一次 commit、wait 的“最近 N 组”条件，以及所有组参加同一轮 CTA 会合。
- 两条日志都是 `depth=2 lag=1 consumers=2`。`gpu_tiling_gemm_tiled_wg_tma` 的 panel group 布局为 `disjoint`，计入原则覆盖。`gpu_tiling_gemm_tiled_wg_tma_h` 的协议也匹配，但布局为 `overlaid`，仍排除。分类器应在这个合作路径检查深度、lag、组数和 WGMMA 批次，再进入现有 `check-layout` 分支。独立生产者走下节的单独分支。


### 独立生产者与 empty mbarrier（第四项，新增 1 个环）

`WarpSpecialized.Fold` 对应单 CTA、两个消费 warpgroup、`lag=1` 的 `ws_fold_at`。生产者和消费者分别推进，`nth i = i`，每槽一对 full / empty mbarrier。初始化后，full 等待比较 `(i/d)%2`，empty 等待比较 `(i/d+1)%2`。每组四个 warp 的首 lane 分别 arrival，empty 的完整相位数由该物理槽已发生的 arrival 总数整除 8 得到。每块数据发布仍需 full arrival 和全部字节。循环内的交还由 `wait_group 1` 之后释放上一块实现，最后一块在退出的 `wait_group 0` 之后单独释放。

- `EmptyBarrier.handshake` 对事件时间作联合归纳。full poll 成功推出该代已发布，empty poll 成功推出每个较早代的八个 arrival 都发生过。只到七个、各组进度不同、奇偶位多次回绕都包含在证明中。
- `WarpSpecialized.read_current_generation` 与 `WarpSpecialized.overwrite_after_last_read` 分别证明 T1、T2。后者沿每个消费组自己的 WGMMA 等待、warp arrival 和生产者的 empty 等待建立顺序。`WarpSpecialized.exists_execution` 在 `d ≥ 2` 给出处理每块并完成末次交还的见证。[握手证明](Ring/EmptyBarrier.lean) · [完整协议](Ring/WarpSpecialized.lean)
- `gpu_tiling_gemm_tiled_wg_ws` 日志为 `depth=4 lag=1 consumers=2`，panel group 布局为 `disjoint`。分类器可把这个非 persistent、非 cluster 的 producer/consumer 配置送入布局检查，核对一个生产者、两个消费组、empty 初值 8、逐 warp 的 arrival、正常及收尾释放。其余 `ws_fold` 配置仍须分别核对。
