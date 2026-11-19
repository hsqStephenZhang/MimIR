# MimIR GPU 工作：贡献清单与开放方向（供逐条核实）

整理日期 2026-11-19，分支 `feat/gpu-experiment-thoughts`，HEAD `f12e45a50`。
相对上游 `origin/master`（`19d6c2de6`）共 514 个提交，`src`/`include` 下 156 个文件、+27386/−2122 行。
GPU 部分从 `d3b950295`（2026-11-07，`docs(gpu): design multi-level tiling for GPU SIMT`）开始；之前约 170 个提交（2026-10-30 至 11-07）是 CPU 侧的图模式匹配、BLIS、自动分组，列在 (f)。

口径：
- 文中的数字全部转引自仓库里的文档，本次没有重新测量；每个数字都给出出处。
- “未核实”指本次没有找到原始记录、没有运行，或文档之间互相矛盾。
- 提交号均已用 `git cat-file -e` 确认存在；lit 文件均已确认存在；lit 是否通过、Lean 是否能构建，本次都没有运行。
- “论文”指 `/home/chzh00001/workspaces/MimIR/cgo27-paper248.pdf`（Principles for Multi-Level Compiler Design）。

## 0. 论文里已有什么（划界用）

| 论文内容 | 位置 | 与本工作的关系 |
|---|---|---|
| `Tileable` 依赖类型接口、`Schedule = Tileable → Tileable`、5 个调度组合子、由 PE 展开成循环嵌套 | §7 | 单个 op、只改迭代空间、单一执行者（`gpu-tiling-formalization` §1 明确指出 GPU 多级 tiling 越过了这三条边界） |
| affine map 是函数、shape 是类型里的项、`map_reduce`、`slice_impl` 由 PE 降级 | §5–§7 | 本工作的基础设施，非新贡献 |
| LoC 对比表（tensor/btensor/affine 对 linalg） | §8.2 表 1 | 无 |
| CPU：KernelBench L3，几何平均 0.87×（直连 LLVM）/1.07×（经 MLIR），Intel Core Ultra 7 265H | §8.3 图 4a | CPU 的 register-blocked `map_reduce` 加 epilogue 是否就是 (f) 里的 BLIS/分组提交，**未核实** |
| GPU：KernelBench L3 对 IREE/CUDA 几何平均 1.70×（更慢），RTX 4060 Ti；只有显存管理与 kernel launch，“kernel fusion 尚未实现”，kernel 数平均 3.5× | §8.3 图 4b | 论文的 GPU 路径就是本仓库的 E0（`gpu.lower_btensor_map_reduce`，一线程一输出；`bench/gpu_tiling/RESULTS.zh-CN.md` “E0” 节）。**GPU 上的多级 tiling、shared/寄存器层级、tensor core、异步协议、融合全部不在论文里** |
| 前端：PyTorch → MimIR 的 Python 绑定 | §8.3 | `mimir-frontend` 尚未适配 GPU 改动（`next-directions` D6） |

表中“论文重叠”一列：**重**＝论文已有；**延**＝延伸论文已有的机制（PE 判定、调度即值）但内容是新的；**无**＝论文没有。

## 1. 贡献清单

### (a) 多级 GPU tiling 与融合

| # | 主张 | 证据 | 状态 | 已知局限 | 论文 |
|---|---|---|---|---|---|
| a1 | `tile_req` 加类型化层级链，层级绑定到 work-group/work-item/warp/warpgroup/cluster/persistent，一份规格经 `TileFuse`→`LowerToMem`→`gpu.outline` 生成 kernel | `c6f5c8667` `955757df7` `b703d9f5e`（CPU 多级）、`5215e36d5`（GPU）、`3acfc3795`（作用域索引链）、`5de3642cb`（五个 nom 合成一个 `TileLevels`）；`src/mim/plug/tensor/phase/tile_fuse.cpp`（1833 行）、`lower_to_mem{,_stage,_tc}.cpp`（共 4746 行）；`docs/gpu-tiling-design.zh-CN.md` §2、§8；`docs/tile-pipeline.zh-CN.md` | 已实现 | 只有静态形状：运行时长度的原型在 `wip/dynamic-extents`（`700e4539f`），2026-11-14 决定推迟（`next-directions` §5 D6）；符号 tile 尺寸回退为不切分（`lit/tensor/tile_symbolic_strict.mim`）；调度靠手写模板（`bench/tritonbench/*.mim` 的 `@占位符@`） | 延 |
| a2 | 输入提升为 shared 面板，协作填充，面板布局（padding、XOR swizzle、消费者感知）作为 Mim 项选取 | `588933131` `f7b31705f` `edb8bfbca` `aacc7accb` `5e360c58b`；`src/mim/plug/tensor/phase/panel_layout.cpp`；`bench/gpu_tiling/RESULTS.zh-CN.md` G2–G3、“float4 与 swizzle” | 已实现 | G1/G3 的 box 合法性与读映射一致性未检查（`gpu-tiling-formalization-gaps` §2） | 无 |
| a3 | 跨线程归约（R8）：work-item 树形合并、warp shuffle、lane 归约 | `868b01cc1` `ab5cb75b2` `19c628fce` `149c91847` `c6ef99a73` `cb46c1dab`；`lit/nvptx/tile_reduce_pair.mim`；行求和 2.61 ms→0.291 ms（1650S，`gpu-tiling-design` §7.1h） | 已实现 | 只有可交换、可重结合的合并；有序树、scan 没有（`gpu-protocol-skeletons-survey-codex` §4.1.1） | 无 |
| a4 | 融合：逐点生产者融进面板填充（dequant GEMM）、归约生产者常驻 work-group（单 kernel softmax/行求和）、epilogue 融合、单 kernel attention | `6bdd8653e`（dequant）、`5adaac246` `f52697047`（常驻）、`f284c8628`（`post_over` epilogue）、`84250830c`（attention 单 kernel）；`bench/gpu_fusion/RESULTS.zh-CN.md` | 已实现 | GPU 上自动分组关闭：`src/mim/plug/tensor/normalizers.cpp:231` 在加载 `gpu` 时返回假，GPU 融合靠显式 `tile_req`；`law.fuse_shared` 的“生产者读消费者暂存行”优化在删除解释器时丢失（`tile-law-design` §13：27.0→29.6 µs） | 无（论文明言 GPU 融合未实现） |
| a5 | warp 代数：warp 值是 `«32; T»` 的纯规格，可在 CPU 求值；分布降级把跨 lane 读换成 shuffle；分片互转、quad 归约、C→A 置换写成引理 | `6bef62d5c` `809c7b2b3` `c0db7a96d` `d50d5b1de` `56b489b9a`；`docs/warp-schedule.zh-CN.md`；`lit/tensor/warp_frag.mim`（`refly.check` 穷举）、`lit/{gpu,nvptx}/warp_lanes.mim`；M3a：手写 vs warp 程序 0.305/0.306 ms（RTX PRO 6000，`warp-schedule` 里程碑） | 已实现（M1–M3b）；M4（TileFuse 自动生成）未核实 | 前提“warp 内收敛”由 outline 检查，不是类型 | 无 |
| a6 | tensor core：TF32/F16/BF16 `mma.sync`、`ldmatrix`（含 B 转置）、Hopper `wgmma`（SS 与 RS）；片段布局与面板布局各只有一份 Mim 定义 | `3698ffc72` `bfa2a4011` `476b05d29` `c599219b1` `654499eb2` `fbb1cd3fe` `dc76f92ea` `835e994a7` `4ec381ffe` `ad65a62c1`（`layout_expr.tc_mma`/`tc_wgmma`）；`a581f48c4` `775348011`（片段坐标由 Mim 布局导出并证明） | 已实现 | 只有 F32 累加；无 FP8；无 Blackwell `tcgen05`/TMEM；TF32 的 B 片段仍是标量读 | 无 |
| a7 | 合法性规则写在 Mim 里由 PE 判定：层级嵌套（`check.walk`）、MMA 形状、共享内存容量、目标指令（`-X nvptx:sm`），拒绝时说明是目标、结果还是本 lowering 禁止 | `f34347640` `9c3226659`（规则写在 Mim、由 PE 判定）、`04daa79ce`（目标）、`2429c4e8c`（容量）、`638bc9c0b`（分类拒绝）、`5de3642cb`（`check.walk`）；`docs/tile-check-rules.md`；`lit/error/target_wgmma.mim`、`tile_pingpong_clipped.mim`、`tile_ws_blocks.mim` | 已实现 | 规则只管“请求是否合法”，不管 lowering 是否忠实；BK=128 五级 240 KB 启动失败而未被拒（`RESULTS` 2026-11-18）；PE 开销与可靠性问题见 D8 | 延（论文的 `if_static`/`ts0` 是同一机制的一行例子） |
| a8 | F₂ 布局判定：同一个 `layout_expr.interpret` 上加 F₂ 索引代数，单射、连续、128B swizzle 一致性（K/MN 主序、TMA 填充）由高斯消元判定 | `4db44630c` `79d4da258` `eb47d1bf2` `d63168417` `2666cafaa`；`lit/tensor/layout_f2.mim` | 部分 | 只对 2 的幂形状；非 2 的幂（尾块、padding）不在 F₂ 里；布局转换（R1）没有 | 无 |
| a9 | TileLaw：把 `map_reduce` 封成带三条切分引理的定律，调度由 Mim 解释器实现 | `ff3b1d648` `305a6a600` `61787947c` `d507a76f1` `ad5c031ff`；`docs/tile-law-design.zh-CN.md` §0–§9 | **解释器已删**（`0d85d5206`）：只剩规格与 `law.req = tile_req`。原因：加载 `plugin tensor;` 12.2 s→1.2 s，`law_wgmma_ws` 13.4→1.6 s（`tile-law-design` §10） | 删后失去 `Plan`/`Acc` 类型对面板所有者、生存期的保证 | 延 |
| a10 | epilogue：16 位结果经带 padding 的 shared 暂存后 16 字节合并写回、TMA 存回；环槽与暂存区叠放使 2048 F16 GEMM 每 SM 两个 work-group | `64b761499` `0d2378c6d` `12c09d0f9`；`bench/gpu_tiling/RESULTS.zh-CN.md` “参考 Triton…”“暂存区与环槽共用” | 已实现 | TMA 存回慢于直接写（逐元素算 swizzle 地址，`STS.U16`）；无 `stmatrix` | 无 |
| a11 | 波数代价模型：`law.best` 按字节数、再按 work-group 波数选候选 | `a883adb56` `600c265b8` `200615538` | 部分 | 只在 law 候选间选；未接入 `tune.py` | 无 |

### (b) 协议与环的降级（cp.async、TMA、WS、cluster、persistent）

| # | 主张 | 证据 | 状态 | 已知局限 | 论文 |
|---|---|---|---|---|---|
| b1 | cp.async 多级环（`-X tensor:pipeline=n`），超过 48 KiB 时改用动态 shared | `9ba7e7708` `a5859d917` `3d927704d`；A100 TF32 4 级 85% cuBLAS TF32（`RESULTS` “多级流水”） | 已实现，A100 实测 | SIMT 路径流水无收益，默认关 | 无 |
| b2 | 协议骨架搬进 Mim：`superstep`、`ring_fold`（`async_ring`/`mbar_ring`/`shared_ring`/`cluster_ring`）、`merge_rounds`、`lane_merge`、`tma_fill`、`tma_ring`、`wgmma_batch`、`merge_pairs`、`persistent_tiles`、`ws_fold` | `f8adec190` `19c628fce` `149c91847` `f8a0db4d7` `df8870e3c` `bfa547cba` `db34e116e` `5fdd5d1de` `ab5006d0e` `b8675ceb5` `a518605c5` `39c3cc8c4`；`src/mim/plug/gpu/gpu.mim`（1260 行）；`tile-law-design` §8.2、§9.7 | 已实现 | 仍在 C++：批之外的协作填充、`lower_tiled`、`gpu.outline`（`gpu-tiling-handoff` §3） | 延（“写在 IR 里由 PE 展开”是论文原则） |
| b3 | TMA + mbarrier 填充、命名 barrier、`setmaxnreg` | `cdf91ceb7`（原语）、`95506e6ab`（tensor 层）、`0ecf2b4d6`；`lit/{gpu,nvptx}/tma.mim` | 已实现，H200 执行 | — | 无 |
| b4 | warp 特化（TMA 生产者 warp + MMA 消费者）、ping-pong | `3c2f78210`；`gemm_tiled_wg_ws.mim`；`lit/error/tile_ws_blocks.mim`（单块拒绝）、`tile_ws_cluster_split.mim` | 已实现 | 编译路径的 attention（`attn_wg`）没有 WS/TMA，只有手写的 `flash_fa3.py` 有（`gpu_attn/RESULTS` FA3 节） | 无 |
| b5 | thread-block cluster：TMA multicast、DSM 上的 split；生产者 warp 可与 cluster 同用 | `3928d6fcb` `c64fbcafe` `197ebdf76` `e691cd3cf`（empty mbarrier 去掉 cluster 定序，0.0192→0.0134/0.0100 ms） | 已实现 | 共享面板在所测尺寸上**没有收益**；128×128、`wg_red_ws` 8 级共享在 2048/4096 慢两个数量级，原因未查（`RESULTS` “cluster 共享面板”） | 无 |
| b6 | persistent work-group：环跨 tile 保持；不再要求每 tile 整圈 | `aa565e27d` `a79ed82c8` | 已实现 | persistent 与 cluster 不能同用（survey §4.1.1）；无 CLC 式取下一项重叠 | 无 |
| b7 | 环的时序参数：`lag`（只等上一步 MMA、交还上一步槽）、`turns`（按环深展开、槽号字面量）、`exact`（不填末块之后） | `bddd1decd` `cdf2252e2` `d76b6fb54` `f566f6a61`；`RESULTS` “有了 lag 之后重新扫调度” | 已实现 | `cdf2252e2` 让 attention 寄存器 112→162 而回退，靠 `turns` 修（`gpu_attn/RESULTS` “回退与修复”）；ptxas 把累加器读排到 `wgmma wait` 前，用 `membar.cta` 绕过（ptxas 12.9/13.1 行为） | 无 |
| b8 | 后端修补：`__nv_fast_*`、ptrmask/align 按 LLVM 版本选、cycle profile | `73b561ff1` `2dd7179cf` `1d823430e` `7cb97a206` | 已实现 | — | 无 |

### (c) 类型化协议与检查

| # | 主张 | 证据 | 状态 | 已知局限 | 论文 |
|---|---|---|---|---|---|
| c1 | 相位类型：work-group 内存令牌 `GM (g, e, p)` 与 `SharedBuf` 按构造、超步、阶段索引；`gpu.erase_phases` 擦除，无运行时代价 | `5aeb1dfb8` `003520de3` `af27098c6`；设计来源 `/home/chzh00001/workspaces/MimIR-refactor/docs/gpu-sync-typed-design-agy.zh-CN.md`（Step 1 部分落地；其 Step 2 的 6 状态 `Slot` 自动机与 Step 3 的 `WM`/`IM` 收敛作用域**未按该文实现**，被 c2 的槽令牌取代或未做） | 部分 | 收敛性（uniformity）没有类型（`linear-types-design` §8） | 延 |
| c2 | 槽令牌 `Held`/`Vacant`，骨架对其参数化；`gpu.check_sync` 在 PE 之后按白名单检查单次使用 | `aeb71f6e4` `9f465cb6f` `3ea4cb084` `12ebdcee2` `8651695d8`；`src/mim/plug/gpu/phase/check_sync.cpp`（505 行）；`lit/error/ring_token_*.mim` | 已实现 | 线性性不在核心类型系统，是 PE 后数使用次数；nominal 包装仍漏检（`tile-law-design` §29） | 无 |
| c3 | 令牌加固：刚性比较、私有构造器、拒绝 Hole/⊥、闭包追踪、环身份 `rid` 进类型、`gpu.Panels (rid, depth)` 句柄、`InFlight` 私有、两相 `gpu.Ring` 状态拒绝同一步两次 `await` | `8c9340acc` `201b2f5ac` `3a36603f3` `b8322375e` `961c8c9a6` `8871654d7` `d4b2e86f7`；`lit/error/ring_await_twice.mim`、`ring_state_alias.mim`、`ring_panels_other_ring.mim`、`ring_token_other_ring.mim`、`ring_in_flight_forge.mim`；`docs/linear-types-design.zh-CN.md` §16–§21 | 已实现 | `ring_idle` 不计次数，从初态两次 `await` 不拒；rolled 环与 `ws_fold` 读 `slots#0` 不经 `read_slot`；槽号相等是 C++ 的模深度算术（`same_slot`）；代（generation）不在索引里（§21.3、§2.3） | 无 |
| c4 | 可执行协议模型写成 Mim、由 PE 穷举：`ring_safe`（TMA mbarrier parity）、`cp_ring_safe`（wait_group），`pipeline_plan` 成为 parity 与 pending 的唯一来源 | `a5d0ab2b7` `5e52f7a53`；`lit/tensor/ring_protocol.mim`、`cp_ring_protocol.mim` | 已实现 | 只对字面量块数做至多 16 步的有界模拟（`linear-types-design` §2.3） | 延 |
| c5 | 类型化数据 tile 外壳 `Tile r unit at life space cap r2 s lay T` 与 buffer 层解释器 | `ce261b719` `bcc8c9a7d` `6f1049e16` `9cdef7c76`；`bench/typed_tiles/`；`lit/gpu/typed_tiles.mim`、`lit/nvptx/typed_tiles.mim`、`lit/error/typed_tile_*.mim`（9 个反例） | 原型，**定为不并入编译器**（`gpu-tiling-handoff` §4） | 编译器的调度不用这个类型 | 无 |
| c6 | 变异评测：15 种协议变异注入 `gpu.mim`/`tensor.mim`，44 对改变了生成代码的组合全部被拒：类型/`check_sync` 6、有界模拟 `ring_safe` 1、Lean 37（其中 1 对是文本格式解析失败）；另有 7 对 `hold_slot` 只改擦除前的幽灵下标，生成代码不变，被类型检查拒（口径见 README §2） | `2ce447bc1` `f12e45a50`；`bench/tools/mutate.py`；`linear-types-design` §22 表 | 已实现 | 只覆盖 7 个环；时序类错误（`early_release`）只有 Lean 能抓；`pending_formula` 不在任何 bench 路径上 | 无 |
| c7 | 差分模糊测试：随机形状与调度对比去掉 `tile_req` 的规格（整数输入逐位一致），发现 3 个静默错误或崩溃 | `bench/validate/fuzz.py`、`README.zh-CN.md`；`lit/nvptx/tile_wgmma_unpipelined.mim`（漏 `fence.proxy.async`） | 已实现（D3 第 1 级） | 不是验证；只在小形状上 | 无 |

### (d) Lean 模型与证书

| # | 主张 | 证据 | 状态 | 已知局限 | 论文 |
|---|---|---|---|---|---|
| d1 | 单 CTA、单消费者 TMA 环的小步模型；T1 `read_current_generation`、T2 `overwrite_after_last_read`、`no_ABA`、`live_read_race_free`、`wrong_panel_rejected`、`exists_drained_iff`（排空当且仅当 `d ≥ 2 ∨ count = 0`）；项目自定义公理为零 | `822bf0149` `db9aabb61` `7a61e3d93` `99e6fa902`；`proofs/ring/Ring/{Machine,Invariant,Fold,Certificate}.lean`；`proofs/ring/README.md` | 已实现（本次未运行 `lake build`，**未核实**） | 物理 kernel 的地址、字节范围、线程收敛、擦除到硬件小步的对应不在证明内（README “与残余 fold 对接”末段） | 无 |
| d2 | 扩展到 `cp.async`、两个消费 warpgroup 的合作 TMA、独立生产者 `ws_fold`（empty mbarrier 奇偶） | `c2c79dd69` `33ad70696` `dbe26e329`；`Ring/{Async,Cooperative,WarpSpecialized,EmptyBarrier}.lean` | 已实现 | cluster（2 个环）、persistent（1 个）、epilogue 叠放的 F16 环（1 个）无模型 | 无 |
| d3 | 可执行检查器及其可靠性（`fold_of_accepted`、`readCode_of_accepted`）；`gpu.ring_trace`（`-X gpu:ring-trace=N`）从 `check_sync` 时的残余抽出逐线程事件；7 个原则覆盖的真实环全部被接受 | `391a25ebd` `82366ff0e` `f14560665` `1176f89a1`；`proofs/ring/{AsyncCheck,CoopCheck,WsCheck}.lean`；`src/mim/plug/gpu/phase/ring_trace.cpp`；`bench/tools/ring_trace_check.py`；`lit/gpu/ring_trace_async.mim`、`ring_trace_tma.mim` | 已实现 | 只解释 `check_sync` 时的残余，之后的 `mem.add_mem`、`affine.lower_for`、后端不在范围；warpgroup 收敛是假设 | 无 |
| d4 | 合成夹具与 Lean `Ring.fold` 逐事件对照 56 个实例；生成器的环按模型分类 7/11 | `9ef7aba65` `f8385cf39` `2942730ec`；`lit/gpu/ring_fold_trace.mim`、`ring_fold_coverage.mim`；`bench/tools/ring_coverage.py` | 已实现 | `lag = 0` 的 28 个实例比模型少一次收尾 `wait_group 0`，语义等价未在 Lean 中证明 | 无 |

Lean 规模：`proofs/ring` 下 21 个 `.lean` 文件共 5340 行，`grep sorry` 为 0。

### (e) 评测基础设施

| # | 工具 | 位置 | 备注 |
|---|---|---|---|
| e1 | cubin 直接加载、同一 context 背靠背计时 | `bench/gpu_tiling/krun.sh`、`kbench*.cpp`；`krun_e2e.sh`（nsys） | 生成的 host 代码每次重建 context，nsys 单次不可靠（`RESULTS` A100 节） |
| e2 | 按层级链计时 F16 GEMM | `bench/gpu_tiling/gemm_levels.sh`（`H=1` 写 F16） | 驱动每次重新分配上传，比 Python 热循环慢（1024：6.45 vs 热态 5.70 µs） |
| e3 | 单 kernel 融合程序 PTX 计时 | `bench/gpu_fusion/ptx_bench.py prepare\|run` | — |
| e4 | tritonbench 算子模板与驱动 | `bench/tritonbench/tb.py`、`tb_driver.cpp` | 依赖 `~/workspaces/tritonbench/.venv` |
| e5 | 规则剪枝的调度枚举 | `bench/tune/tune.py` | 无代价模型，排序靠实测 |
| e6 | 差分模糊 | `bench/validate/fuzz.py`、`one.py` | 同 c7 |
| e7 | 全部 GPU bench 编译、设备 IR 对比 | `bench/tools/compile_all.sh`、`cmp_device_ir.sh`、`canon_ir.py` | `canon_ir.py` 把元数据引用全归成 `!N`，会掩盖 alias 差异（`tile-law-design` §29） |
| e8 | 环覆盖、证书检查、变异 | `bench/tools/ring_coverage.py`、`ring_trace_check.py`、`mutate.py` | — |
| e9 | 重构不变性快照 | `bench/typed_tiles/tools/{snap.sh,strip.py,devsnap.sh,ctime.py,mklit.py}` | — |
| e10 | 周期计数 | `-X nvptx:profile=1`（`7cb97a206`） | — |
| e11 | 仓库外 | `~/workspaces/gemm-bench-gpu-handoff`（22 例 GEMM，未提交，按决定不提交）、`~/workspaces/cutlass-cmp`、`~/workspaces/fw-cmp/REPORT.zh-CN.md`（Triton/TileLang/TK） | 不随 `git push` 走 |

### (f) 前端与图层

| # | 主张 | 证据 | 状态 | 已知局限 | 论文 |
|---|---|---|---|---|---|
| f1 | 图模式匹配库：独立 `PatternProgram`、`GraphView`、多根重写、`compile.rule_app`/`graph_rule_app`，覆盖 TVM DPL 与 MLIR PDLL 参考例 | `edc772d08` `860841a05` 起约 30 个提交；`include/mim/pattern/`、`src/mim/pattern/`；`test/pattern_p5.cpp`；`docs/graph-pattern-matching-design.zh-CN.md`、`graph-pattern-reference-tests.zh-CN.md` | 已实现（P0–P5） | 无通用依赖类型实例化求值、无 PDLL parser；GCC 15.2 在 `graph.cpp` 上 ICE（handoff §1） | 无 |
| f2 | CPU 自动分组、scoped 规则、BLIS 形状打包 GEMM、conv halo、persistent 预打包，默认开启 | `7543cf899` `b85954aa6` `485f1b55e` `a74cd2dcd` `db4055a84`；`docs/graph-pattern-tile-fuse-experiments.zh-CN.md` Task 19A.3 表；`docs/row-pipeline-benchmark.zh-CN.md` | 已实现（CPU） | GPU 上关闭（`normalizers.cpp:231`）；2048³ 仍慢于 OpenBLAS | 可能与论文 §8.3 CPU 部分重叠，**未核实** |
| f3 | attention 用图规则 raise、online 规则选择（CPU） | `b07049508` `aba007b2e` `7cc84d130` `883bdae8c7` `181a6db1e4` `921dd02fde`；`docs/attention-schedule-experiment.zh-CN.md:208-1374` | 已实现并测量（步骤 1–5 都已完成；第 157 行那节“步骤 1 阻塞”是中途记录，汇总时依据 agy 核对更正）。负面结果：比不启用规则慢 1.43–3.11×（attention）、1.19–2.46×（transformer），堆分配 3L²+132L+87 次（`:1273-1360`）；GPU 上关闭 |
| f4 | 真实模型（`models-3137f3dbfd`，seq 128/512/1024）的调度诊断：LayerNorm 重算、Linear 未得到 BLIS 的原因 | `docs/model-scheduling-diagnosis.zh-CN.md`；`298841389` `32cea4feb` | 诊断加部分修复（CPU） | — | 无 |
| f5 | PyTorch → MimIR → GPU 的端到端 tiled 路径 | — | **未实现**：`mimir-frontend` 未适配 GPU（`next-directions` D6），GPU bench 都是手写 `.mim` 模板 | — | 论文的前端已有（E0 路径） |

## 2. 性能汇总

所有数字转引自文档；“比例”＝基线时间/我们的时间（>100% 表示更快）。

### 2.1 GEMM

| 硬件 | 精度 | 规模 | 我们 | 基线 | 比例 | 出处 |
|---|---|---|---:|---:|---:|---|
| H200 NVL sm_90a | F16 输入/F16 输出，wgmma+TMA | 1024³ | 6.22 µs | cuBLAS 6.02 | 96.8% | `bench/gpu_tiling/RESULTS.zh-CN.md` “参考 Triton、TileLang、ThunderKittens 之后” |
| 同上 | 同上 | 2048³ | 27.30 µs（叠放后） | cuBLAS 25.68 | 94% | 同文件 “暂存区与环槽共用” |
| 同上 | 同上 | 4096³ | 199.97 µs | cuBLAS 197.6–198.9 | 99% | 同上 |
| 同上 | F16 输入/F32 输出 | 8192³ | 1689 µs | cuBLAS 1692 | 100% | 同文件 “有了 lag 之后” |
| 同上 | 热态计时同一 cubin | 1024³ | 5.70 µs | Triton 5.85、cuBLAS 6.02 | 快于两者 | 同文件 “暂存区与环槽共用”；`~/workspaces/fw-cmp/REPORT.zh-CN.md` |
| RTX PRO 6000 sm_120（mma.sync） | FP32/TF32/F16/BF16 | 1024–4096³ | 见出处 | cuBLAS | 72–89% | `RESULTS` “非 wgmma 路径对 cuBLAS · 调过 tile 之后” |
| A100-PCIE sm_80 | TF32 mma.sync，4 级 cp.async | 4096³ | 1.87 ms | cuBLAS TF32 1.584 | 85% | `RESULTS` “多级流水与动态 shared memory” |
| A100 | FP32 SIMT 128×128 | 4096³ | 9.28 ms | cuBLAS FP32 7.256 | 78% | `RESULTS` A100 节 |
| RTX 3090 sm_86 | FP32 SIMT，22 例 | 混合 | — | PyTorch | 几何平均 52.4%（TVM 50.9%）；大方阵 72–85%；小 M 9–18% | `~/workspaces/gemm-bench-gpu-handoff/results/gpu-tiling-ss420e-20261116/comparison.md`（仓库外） |
| GTX 1650S sm_75 | FP32 SIMT，22 例 | 混合 | — | PyTorch | 几何平均 87.0% | 同目录 `gpu-tiling-local-20261116/comparison.md`（仓库外） |
| H200 | addmm（bias，F16） | (20120,512,1536) | 0.0805 ms | aten 0.0608、inductor 0.0772 | 76% / 96% | `bench/tritonbench/RESULTS.zh-CN.md` |
| H200 | linear+gelu | (8192,8192,2048) | 0.552 ms | aten 0.529 | 96% | 同上 |

注意：
- `next-directions` §1 的“GEMM 为 cuBLAS 的 85–91%”是 2026-11-13 写 F32 的旧口径（4096³ 0.2247 vs 0.204 ms，event 时间），已被上表 2026-11-17/18 的 kernel 时间口径取代。
- 对 CUTLASS：只有逐项分解（主循环斜率我们 0.049 µs/k，CUTLASS 128×128 0.051–0.054，cuBLAS 0.0448；4096 上 F32 直接写回的固定开销 23.6 µs vs CUTLASS TMA 存回 F16 5.8 µs），没有端到端比例表（`RESULTS` “与 CUTLASS 逐项对比”）。

### 2.2 Attention

| 硬件 | 实现 | 形状 | 我们 | 基线 | 出处 |
|---|---|---|---:|---:|---|
| H200 | **编译路径** `attn_wg`（F16，wgmma，cp.async 双缓冲，无 WS/TMA） | 32×2048，D=64 | 0.121 ms | cuDNN 0.098、PyTorch FA2 0.145 | `bench/gpu_attn/RESULTS.zh-CN.md` “回退与修复” |
| H200 | 同上 | 128×4096 | 1.747 ms（11-13） | cuDNN 1.283（0.73×）、手写 `flash_wg` 1.691（0.97×） | 同文件 “编译路径：warpgroup 拥有的区域” |
| H200 | **手写** FA3 结构（WS+TMA+mbarrier+setmaxnreg+ping-pong） | 128×4096 | 1.401 ms（392 TF） | cuDNN 1.283 | 0.92×；同文件 FA3 节 |
| H200 | 编译路径 `attn_warps`（TF32 mma，单 kernel）对 PyTorch 四 kernel | L=4096，H=12 | 495.7 µs | PyTorch TF32 1555、F16 1136 | 314%/229%；`bench/gpu_tiling/RESULTS.zh-CN.md` 末节（注意：对照的是非融合的 PyTorch，不是 FlashAttention） |
| RTX PRO 6000 | `attn_law` vs `attn_warp` | 128×2048 | 1.042 vs 1.046 ms | — | `gpu_attn/RESULTS` 块定律节 |

结论：编译出的 attention 只比手写同结构慢约 3%，但离 cuDNN/FA3 还有 0.73–0.88×；编译路径没有 warp 特化与 TMA。

### 2.3 访存密集

| 硬件 | 算子 | 我们 | 基线 | 出处 |
|---|---|---:|---:|---|
| H200 | softmax 4096² F16 | 0.0259 ms（调度搜索 + fast exp） | torch.compile 0.043–0.055 | `bench/tritonbench/RESULTS.zh-CN.md` 第四轮 |
| H200 | cross_entropy 16384×65536 | 0.984 ms | torch.compile 2.037（`next-directions` §1） | 同上 |
| H200 | rms_norm / layer_norm / swiglu | 0.0102 / 0.0104 / 0.0474 ms | torch.compile 0.0131 / 0.0168 / 0.0513 | 同上，第一节 |
| H200 | softmax 12×1024×1024 F32 | 26.9 µs（`softmax_resident_shared`，现已不可复现，见 a4） | PyTorch kernel 30.3 | `bench/gpu_tiling/RESULTS.zh-CN.md` 末节 |
| H200 | 行求和 | 14.0 µs | PyTorch 16.5 | 同上 |
| RTX PRO 6000 | softmax / 行求和 | 20.6 / 9.9 µs | 22.7 / 17.3 | `bench/gpu_fusion/RESULTS.zh-CN.md` sm_120 节 |

### 2.4 CPU（非 GPU，供划界）

512³ GEMM+bias+ReLU 约 95.4 GFLOPS，默认分组比 `group=0` 快 2.6–3.3×，2048³ 仍慢于 OpenBLAS（`docs/graph-pattern-tile-fuse-experiments.zh-CN.md` Task 19A.3）。

### 2.5 没有的数字

- 端到端模型（KernelBench、transformer block）在新 GPU 路径上的结果：没有。
- Blackwell 数据中心卡（tcgen05/TMEM）：没有硬件。
- 带流水环配置（`mimir-tiling128p3`）的 22 例全套：没跑成（handoff §4）。
- 类型检查与 `ring_trace` 的编译时间开销：只有 `rid` 改动“设备 IR 不变”，没有时间数字；删解释器前后的时间见 a9。

## 3. 开放问题与方向（按论文价值排序，已去重）

> 汇总时的注记：下表第 1 项（核心线性性、环/槽/代进索引、删 `same_slot`）已被 `linear-types-design` §14、§17 和 `reviews/ringstate_report.md` 否决，改走两相状态加残余验证。方向排序以 [README §4](README.zh-CN.md) 为准。

| 序 | 方向 | 现状与缺口 | 来源 |
|---|---|---|---|
| 1 | **协议类型作为编译器输出的验证**：T1 核心线性性（`core.check_linear`）、T2 环/槽/代进索引（删 `same_slot`）、T3 共享区域 typestate、`ring_idle` 只由 `open` 产生、`InFlight` 接到 `release` | 已有：rid、Panels、两相 Ring、白名单检查、变异 51/51。缺：线性性仍是 PE 后计数；代不在索引；rolled 环读 `slots#0`；叠放区域靠人工 fence | `linear-types-design` §4–§6、§11、§21.3；`ringstate_report` §4 |
| 2 | **Lean 覆盖补全**：cluster（multicast、远端 arrival）、persistent（跨 tile 相位）、叠放区域；证明 `lag=0` 收尾等待的等价；把地址/字节/收敛从 TCB 中缩小 | 7/11 环有证书 | `proofs/ring/README.md` 覆盖表；`linear-types-design` §20.3 第 4 项 |
| 3 | **lowering 的翻译验证**（D3 第 2 级）：Mim IR 层符号执行等价检查 | 只有第 1 级模糊；`LowerToMem` 约 4700 行是最大的可信基 | `next-directions` D3；`gpu-tiling-formalization-gaps` G12 |
| 4 | **F₂ 布局代数推广**（D1）：非 2 的幂、布局转换 R1、`Frag` 作为累加器类型 | 2 的幂上的判定已做 | `next-directions` D1、§4；`gpu-tiling-formalization` §7 |
| 5 | **规则即约束的调度合成**（D4） | `tune.py` 只枚举；`law.best` 有波数模型 | `next-directions` D4；`bench/tune/README.zh-CN.md` |
| 6 | **覆盖面**（D6）：前端接 GPU、transformer block 端到端、反向 attention、动态形状、FP8 | 动态形状 2026-11-14 推迟；阻塞在 core 对等价依赖形状的统一 | `next-directions` D6、§5 |
| 7 | **编译路径的 attention 补 WS/TMA/ping-pong**，追 FA3/cuDNN | 手写 0.92×、编译 0.73–0.88× | `gpu_attn/RESULTS` |
| 8 | **GEMM 余差**：环按下标寻址（约 0.1 µs/步）、`stmatrix` epilogue、双缓冲子块 epilogue、cluster 共享面板的收益、小 M（split-K 对 M=49 负优化） | 4096/8192 已持平 | `RESULTS` “与 CUTLASS 逐项对比”“暂存区…”；handoff §4 第 5 项 |
| 9 | **收敛性类型**、参与者类型（部分 CTA 当全组会编译失败） | 不做于本阶段 | `linear-types-design` §8；`tile-law-design` §29 |
| 10 | **Blackwell / 目标描述**（D5）：自建 vs 接 NVIDIA Tile IR | 未决定；无硬件 | `next-directions` D5 |
| 11 | 算法与调度联合改写（D7）、PE 可靠性（D8） | D7 只做了 fast exp；D8 修了隐式参数递归 | `next-directions` D7、D8、§5 |
| 12 | 三套类型打通：调度的索引、外壳 `Tile`、`gpu.SharedBuf` | 解释器删后，`Tile` 只剩显式调度 DSL 在用 | handoff §4 第 2 项；`tile-law-design` §12 末 |

### 3.1 已决定不做或已撤回

| 决定 | 原因 | 来源 |
|---|---|---|
| Mim 里的调度解释器（TileLaw 的 `build_post`）删除 | 编译时间 12.2 s→1.2 s；`Levels` 接得住全部请求（1032 lit 不变） | `tile-law-design` §10 |
| buffer 层解释器不并入编译器，不追剩余差距 | 原型价值已得，代价高 | `gpu-tiling-handoff` §4 |
| 动态形状推迟 | core 重建时依赖形状的 coerce 问题需先定方案 | `next-directions` §5（2026-11-14） |
| 线性类型“缓一缓、用检查替代” | 当时判断结构化组合子足够；2026-11-19 又重新起草（§3 第 1 项），属**改变了的决定** | `next-directions` D2；`linear-types-design` |
| `RingState` 按计数索引（依赖迭代号的循环状态） | `affine.For` 状态类型固定；改用两相 typestate + 残余验证（方案 D′+E） | `linear-types-design` §17；`ringstate_report` §0 |
| rolled 环改成动态下标读以便走 `read_slot` | 会改变设备 IR | `linear-types-design` §21.3 |
| 每步 `bar.sync` 换成 empty mbarrier 到达 | 1024 上 8.20→9.95 µs，leader 拖住 warpgroup | `RESULTS` “与 CUTLASS 逐项对比” |
| `elect.sync` 选发 TMA 的 lane | 时间不变（6.45→6.58） | `RESULTS` “参考 Triton…” |
| BK=128、尾部重填 | 不更快；主循环受 L2 带宽限制 | `RESULTS` 同节、“暂存区…”；`2a7fb6d82` |
| 提交基准仓库 `gemm-bench-gpu` 的改动、移动本地 `feat/gpu-tiling` | 用户决定 | handoff §4 |
| 手选调度与代价模型“暂时不管” | 先做 D1–D3 | `next-directions` D4 |
| 自动化/搜索可能整体砍掉，论文以 GPU 协议类型为中心 | 研究重心决定（用户记忆，非仓库文档，**未核实**于仓库） | `linear-types-design` §15 的定位与之一致 |

## 4. 核实清单

所有命令从 `/home/chzh00001/workspaces/MimIR-thoughts` 运行。按用户约定：不要构建 `lit` 目标（并发过高会 OOM），先构建 `mim_lit_tests` 暂存，再以 `-j8` 跑。

| 主张 | 核实方法 |
|---|---|
| GPU 工作的起点与范围 | `git log --oneline origin/master..HEAD \| wc -l`（514）；`git show --stat d3b950295 5215e36d5` |
| 论文的 GPU 路径是 E0 | 读论文 §8.3 图 4b 段；`bench/gpu_tiling/RESULTS.zh-CN.md` “E0” 节；`git grep -l tile_req origin/master -- src`（应为空） |
| 层级链与合法性规则 | `src/mim/plug/tensor/tensor.mim` 中 `TileLevels`、`check.walk`；`docs/tile-check-rules.md`；`./lit ../build/lit -j8 -a --filter 'target_wgmma\|tile_ws_blocks\|tile_pingpong_clipped'`（在 `lit/` 下） |
| 协议骨架在 Mim | `grep -n 'ring_fold\|ws_fold\|tma_ring\|wgmma_batch\|persistent_tiles' src/mim/plug/gpu/gpu.mim` |
| 槽令牌、rid、Panels、两相 Ring 的拒绝 | 运行 `lit/error/ring_*.mim`（`--filter ring_`）；读 `src/mim/plug/gpu/phase/check_sync.cpp` |
| F₂ 判定、协议模型 | `--filter 'layout_f2\|ring_protocol\|cp_ring_protocol'` |
| Lean 定理与零公理 | `cd proofs/ring && export PATH=$HOME/.elan/bin:$PATH && lake build`，再按 README “构建与复核”跑 `#print axioms`；`grep -rn sorry proofs/ring`（应为空） |
| Lean 与生成代码对照 | `lake exe ring_trace 3 7 1 2`；`--filter 'ring_fold_trace\|ring_trace_async\|ring_trace_tma\|ring_fold_coverage'` |
| 7/11 环覆盖 | `(ulimit -v 16000000; bench/tools/compile_all.sh /tmp/ring-bench -X tensor:ring-coverage=1)`，再 `python3 bench/tools/ring_coverage.py /tmp/ring-bench` |
| 变异 44/44（另 7 对幽灵下标） | `bench/tools/mutate.py <out>`，对照 `linear-types-design` §22 表 |
| `rid` 等改动不改设备 IR | `bench/tools/cmp_device_ir.sh`（两棵构建树），或 `canon_ir.py` 前后对比 |
| H200 GEMM 96.8/94/99/100% | `nvidia-smi` 确认是 H200；`H=1 bench/gpu_tiling/gemm_levels.sh …`（参数见脚本，**未核实**用法）；cuBLAS 用 `torch_kern.py`；对照 `RESULTS` 的调度列（1024：64×128、4 级；4096/8192：128×256、3 级） |
| sm_120 GEMM 72–89% | RTX PRO 6000 上 `bench/gpu_tiling/krun.sh`，链见 `RESULTS` “调过 tile 之后”表 |
| attention 数字 | `BM=128 BK=64 DRIVER=attn_driver16.cpp bench/gpu_attn/kattn.sh attn_wg.mim 32 2048 64 -X nvptx:sm=90a`；FA3 手写：`MIMFLAGS=… GEN=flash_fa3 bench/gpu_attn/kflash16.sh 128 4096 f16 4 reg ov pp` |
| softmax 0.0259 ms | `~/workspaces/tritonbench/.venv/bin/python bench/tritonbench/tb.py softmax M=4096 N=4096 R=1 I=16 P=0` |
| 3090/1650S 22 例几何平均 | `~/workspaces/gemm-bench-gpu-handoff/results/*/comparison.md` 与 `NOTES.md`（仓库外，未提交） |
| Triton/TileLang/TK 对比 | `~/workspaces/fw-cmp/REPORT.zh-CN.md` §0–§1 |
| 删解释器的编译时间 | `git show 0d85d5206 --stat`；`tile-law-design` §10 的命令（`time build/bin/mim` 只含 `plugin tensor;` 的文件） |
| GPU 上自动分组关闭 | `src/mim/plug/tensor/normalizers.cpp:231` |
| agy 设计的落地程度 | 对照 `/home/chzh00001/workspaces/MimIR-refactor/docs/gpu-sync-typed-design-agy.zh-CN.md` §7 四步与 `gpu.mim` 中的 `GM`/`SharedBuf`/`Held` |

### 4.1 本次发现的、需作者留意的不一致

- `docs/gpu-multilevel-first-principles.zh-CN.md` §1.9 仍说 `run`、三种 `kind` 由 “`mod law` 的解释器” 实现；该解释器已在 `0d85d5206` 删除。
- `docs/tile-pipeline-pseudo.zh-CN.md` §3 同样描述已删除的解释器。
- `linear-types-design` 有两个 §19；§19.1 说“满足定理全部前提的为 0”，§20.2/§21.2 改为 7/11，以后者为准。
- `next-directions` §1 的性能表是 2026-11-13 的旧口径（见 §2.1 注）。
- handoff 写在 `feat/gpu-tiling-refactor` 上，lit 结果是 sm_75 上 981 通过、12 个需 sm_80 的测试失败；之后的文档在 H200 上记 1032/1053 个全通过。最新全量数字以哪一次为准，**未核实**。
