# 002c: Stage 3 (arm B, LP-DS as released, Hopper), written 2026-10-30 before any LP-DS run

- Label: REPRODUCTION (released LP-DS code and default config, our pin) compared against arm A (stage 2).
- Arm B: third_party/lpds pinned commit 6ac778a + patch 0004 (logging only), `cfg/gym/lpds_hopper.yaml` unchanged (target_l2_deviation 0.02 = the released default; the paper table's 0.35-0.50 is unresolved, audit.md), Hopper, 200,000 raw steps (`total_timesteps=50000` chunk-steps; verified on a laptop smoke test that eval timestamps count the same way as DSRL's), `eval_interval=1000`, seeds 10-19, 5 concurrent.
- Comparator: arm A from stage 2 (same seeds 10-19, same eval_interval, same step budget). Arm A is NOT re-run.
- Checkpoint: LP-DS's shipped Hopper checkpoint and normalization have the same sha256 as DSRL's (9a5839d3..., d05b2943...); no substitution needed.
- Metric: mean eval reward of the last 3 evals (as 002b). Primary: paired-by-seed-index bootstrap CI of B - A (10,000 resamples, seed 20261030). B better if lower bound > 0; A better if upper bound < 0; else no detectable difference.
- Known mismatches (not matched, reported as caveats): LP-DS uses a deterministic actor, no entropy tuning, Polyak-averaged acting actor, and its own evaluation function (same 25 eval envs / 8 episodes per env, same reward averaging by inspection, not verified identical). H4's tuned-DSRL (arm D) and tuned-LP-DS (arm E) comparisons are NOT run in this stage, so this stage can only speak to "as released vs as released" on Hopper, not to whether LP-DS's advantage survives matched tuning.
- Budget: ~2 waves of 5 runs, expected <= $2 incl. setup. Hard limit/auto-stop as in earlier stages.
