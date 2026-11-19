# Review request: a survey of related work, our contributions and future directions (MimIR GPU work)

You are an expert in ML compilers (MLIR, IREE, TVM, Triton, CUTLASS, TileLang), GPU programming (Hopper TMA, mbarrier, WGMMA, warp specialization) and type systems and verification (linear and typestate types, translation validation, Lean).

This is a **read-only** review: do not modify, commit or push anything. Answer in Chinese.

## What to review
The worktree `/home/chzh00001/workspaces/MimIR-thoughts`, branch `feat/gpu-experiment-thoughts`. The documents are in `docs/thoughts/`:
- `README.zh-CN.md`: the index, the overall conclusions, the cross-document corrections, the merged claims and the directions;
- `A-infrastructure.zh-CN.md`: production and research compiler infrastructure;
- `B-graph-and-dsl.zh-CN.md`: graph compilers versus kernel DSLs;
- `C-tile-and-fuse.zh-CN.md`;
- `D-types-and-verification.zh-CN.md`;
- `E-contributions-and-directions.zh-CN.md`: our own contribution inventory with evidence.

The documents were written by several agents; the author will use them to check the contributions and to choose future directions.

You may check claims against:
- the local clones under `/home/chzh00001/workspaces/`: `llvm-project/mlir`, `iree`, `tvm`, `triton`, `tilelang`, `cutlass`, `ThunderKittens`, `modular`, `cutile-python`, `cubek`, `cuda-oxide`, `oneDNN`, `mimir-frontend`;
- the code and the git history of the worktree itself;
- `docs/linear-types-design.zh-CN.md` and `proofs/ring/`;
- the paper text `/tmp/claude-7076018/-home-chzh00001-workspaces-MimIR/acdcdd87-d117-42ff-af4a-7559329fd915/scratchpad/cgo.txt`;
- the web.

Do not build anything and do not run the test suite.

## Questions
1. **Factual errors.**
   - Spot-check at least 15 cited `path:line` claims across A–E, and every claim the index makes about novelty.
   - List each one that is wrong, outdated or misattributed, with the correct fact and your evidence.
2. **Overstated or unsupported claims**, about our work or about other systems. Say which wording to use instead.
3. **Missing related work** that a reviewer at PLDI, CGO or ASPLOS would expect for each dimension, including recent 2025–2026 work: automatic warp specialization, TMA/mbarrier modelling, linear or typestate GPU APIs, and verified or validated ML compilers. Give references, and say how each changes our claims.
4. **Inconsistencies** between the five documents and the index, and between the documents and the code.
5. **Judgement.**
   - Is the merged contribution ranking (index §3) right?
   - Is the direction ranking (index §4) right?
   - What would you change to make the strongest paper, given that the new multi-level GPU path is not yet connected to the PyTorch frontend?

Rank the findings by severity. Ground each in `file:line` or a citation, and mark speculation as such.
