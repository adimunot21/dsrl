# Amendment 001d: cloud grid protocol

Date written: 2026-10-28, before any grid result other than the outside/0.15/0.2 runs already reported (session 1 at target -dim; cloud benchmark at target 0.0).
Label: SIMPLIFIED MECHANISM TEST.

1. C_N^box (001c item 3) is implemented by **clipping** prior noise to [-b_W, b_W] (b_W = 1.5), not rejection. Clipping puts mass on the box edge, which slightly favours edge noises; this is stated as a limitation. Secondary predictor only.
2. Grid order and jobs are fixed in `scripts/grid_cloud.py` (committed with this note): phase A = 27 configs at target entropy 0.0; phase B = 9-config subset at target -2.0. 36 jobs, lr 3e-4 (chosen in 001b), seeds 0-9, 50k env steps, OMP_NUM_THREADS=4, RTX 3090.
3. The outside/0.15/0.2 benchmark result at target 0.0 lacks C_box and is re-run inside the grid for consistency; its first run is kept as `_nobox` and reported as a replicate of the gain (cross-run determinism).
4. Spend guard: the orchestrator stops launching jobs at $15 estimated (user-reported $0.5/h).
5. H2 (bound sweep) is not in this grid; it gets its own amendment after the H3 primary test is evaluated.
