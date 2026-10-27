# Running log

## 2026-10-27
- Decisions (user): cloud cap ~$100, laptop-first; scope toy + Gym + Robomimic Can/Square; third-party code as pinned submodules, never vendored; open-ended timeline; use parallel agents.
- Phase 0 audit done; see docs/audit.md. No spend, no pushes, no downloads.
- Surprise: all four issue #6 claims hold in code, but the double denoising is limited to warm-up rollouts.
- Surprise: LP-DS's trust-region quantity is per-dimension-normalized in code, so its delta is not the paper's delta; deltas in the repo differ from the paper's table for Gym tasks (unconfirmed).
- Process note: subagent reports are unverified until I check the cited lines. 7 of the claims above I re-checked in code.
- User decisions (2026-10-27): skip the license query to the DSRL authors; public Drive checkpoints may be used (non-commercial, cite the papers). We still keep third-party code as pinned submodules, not copied into our tree.
