# 002b: Stage 2 (dense-eval replicate of A vs C), written 2026-10-30 before any stage-2 run

Motivation: stage 1 (A vs C, Hopper, seeds 0-9) gave C - A = -144 [-248, -38], opposite to H5, but with eval_interval=4000 calls (16,000 chunk-steps) each run had only 4 evals, so the "last 10%" metric was one noisy eval. Stage 2 asks whether the deficit of C survives a less noisy metric on independent seeds.

- Label: REPRODUCTION / NEW EXTENSION (same arms, new seeds, denser evaluation).
- Arms A and C unchanged (patches as in configs/gym_budget.json). Hopper only. 200,000 raw steps.
- Seeds 10-19 (fresh; not tuned on, not used in stage 1). Stage-1 seeds 0-9 are not pooled in the primary test; a pooled 20-seed estimate is reported as secondary.
- Change: `eval_interval=1000` (calls; = 4,000 chunk-steps), giving ~13 evals/run. Learning is unaffected (evaluation only).
- Primary metric: mean eval reward over the last 3 evals of each run (replaces "last 10%", which is 2 evals at this density; fixed now to reduce noise). Also report mean over all evals after step 0 (area-under-curve proxy).
- Primary test: paired-by-seed-index bootstrap CI of C - A (10,000 resamples, seed 20261030).
- Decision: "C worse than A" is replicated if the CI upper bound < 0; "no difference" if CI contains 0; "C better" if lower bound > 0. Whatever occurs is reported; no redefinition afterwards.
- Budget: ~2 waves x ~1.2 h at $0.507/h, <= $3 including idle; watchdog hard limit applies. Auto-stop after verified completion.
