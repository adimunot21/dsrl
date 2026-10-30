# How much can a frozen noise-controllable policy gain from learned initial noise, and when does steering fail?

Final report, 2026-10-31. Simulation only. Every number below traces to a file in `results/` and a decision in `docs/prereg/` or `LOG.md`.
Experiment labels: **REPRODUCTION** (released code, our pins), **SIMPLIFIED MECHANISM TEST** (toy decoders we built), **NEW EXTENSION** (our own design).

## 1. Scope actually delivered (and what was cut)
The brief assumed a ~$100 cloud cap, three Gym tasks, six arms and Robomimic. After the toy grid the remaining budget was ~$10, so the Gym phase was cut:

| Planned | Delivered |
|---|---|
| Toy decoders, probe, H1-H3 | Done: 27-config grid x 10 seeds (+ 9-config entropy subset), pre-registered analysis |
| Gym: Hopper, Walker2d, HalfCheetah; arms A-G | **Hopper only; arms A, B, C**. No Walker2d/HalfCheetah, no tuned arms D/E, no arm F (clip), no arm G (entropy) |
| Robomimic Can/Square | **Not run** (gated on cost; the measured Gym throughput made it infeasible at this budget) |
| >= 10 seeds | Arm A: 10 (seeds 0-9, stage 1) + 10 (seeds 10-19, stage 2); arm C: 10 + 10; **arm B: 8 complete** (2 cut off by a pod time limit) |
Cloud spend: toy grid ~$10 (user-reported); Gym stages 1-3 ~ $7-8 (pod rates $0.507/h confirmed for stages 1-2; the stage-3 A40 rate was never confirmed, assumed <= $0.5/h). Roughly $17-18 in total against the original cap, none of it beyond what the user authorised.

## 2. Findings

### 2.1 Toy decoders (SIMPLIFIED MECHANISM TEST; prereg 001, 001a-d)
Setup: frozen decoders over a 2-D multi-goal task with tunable support (inside/edge/outside), concentration and weakness; DSRL-SAC-style noise steering (batched 10-seed ensembles); a forward-pass-only **search ceiling** probe C_N (best of N noises per state).
- **H3 (probe vs entropy as predictor of steering gain).** Spearman rho across 27 configs: probe headroom H_probe_256 = **0.736** [0.420, 0.912]; kNN action entropy = 0.409 [-0.04, 0.75]. Supported by the letter of the pre-registered rule (rho >= 0.7 and CI lower bound > entropy's point estimate) but the margin is thin (0.420 vs 0.409) and the CI is wide: **weak evidence, not a validated predictor**. Kill rule (rho < 0.5) not triggered. Exploratory only: within the "weakness" strata, entropy matched or beat the probe at weakness 0.2 and 0.5 (0.90 vs 0.77; 0.95 vs 0.85) and the probe was negative at weakness 0.05.
- Secondary, post-hoc-labelled: the box-restricted ceiling (prior noise clipped to the steering bound) correlates at 0.944 [0.822, 0.99]; action variance 0.499. These did not enter the primary decision.
- **H1(i) (gain bounded by the ceiling).** 0 of 270 (config, seed) runs exceeded C_ref + 0.05: passes (limit 5%).
- **H1(ii) (no gain when the optimum is outside the base support): NOT SUPPORTED.** On 6 unreachable configs the mean gain was 1.163 [0.979, 1.330] vs 1.248 inside. The reward is dense, so steering gets partial credit toward unreachable goals; the pre-registered expectation was wrong and was not redefined after the fact. A sparse-reward re-run (which would test reward hacking too) was not done.
- **SAC entropy target sets an effective noise bound.** With target entropy 0.0 (DSRL's setting) vs -dim, steering stalls short of the optimum; in a linear-decoder check the gap to the analytic optimum was -0.34 at target 0.0, -0.11 at -2, -0.013 at -6, ~0 at -20 (`results/entropy_bound_test_*.json`). In the 9-config subset a target of -2 beat 0.0 in 9/9 configs (+0.01 to +0.09 gain). This is a mechanism finding in the toy; it was not tested on Gym (arm G not run).
- Probe values were identical across laptop and cloud hardware (determinism check).

### 2.2 Gym, Hopper (REPRODUCTION / NEW EXTENSION; prereg 002, 002a, 002b, 002c)
All runs: 200,000 raw env steps (= 50,000 chunk-steps), base policy and normalization verified by sha256 against `docs/checkpoint_manifest.json`, bootstrap CIs over seeds (10,000 resamples), full per-seed values in `results/gym_stage*_analysis.json`.

| Stage | Comparison | Metric | Result |
|---|---|---|---|
| 1 (seeds 0-9) | A vs base policy | final eval (4 evals/run) | A 1908 [1826, 2002]; gain +292 [211, 379] over the base policy (~1620) |
| 1 | C - A | final eval | -144 [-248, -38], C > A in 2/10 |
| 2 (seeds 10-19, 13 evals/run) | A | mean last 3 evals | 1808 [1758, 1868]; gain ~ +200 over base |
| 2 | **C - A** | mean last 3 evals | **-5 [-72, +56]**, C > A in 7/10: **no detectable difference** |
| 3 (seeds 10-17, n=8) | B (LP-DS as released) vs A | mean last 3 evals | A 1824 [1763, 1894]; **B 1591 [1584, 1598]**; **B - A = -233 [-302, -173]**, B > A in 0/8 |

- **DSRL-SAC reproduces qualitatively on Hopper:** every one of 20 seeds ends above the base policy. We did *not* compare against the paper's reported numbers, and 200k raw steps is a small fraction of the released default budget (hard-coded 20M).
- **The truncation-fix arm (C).** The stage-1 deficit (-144) did not replicate on independent seeds with denser evaluation (-5, CI containing 0). Stage 1 used a single final evaluation per run, which we flagged as noisy in advance. Net: **no demonstrated effect of the time-limit/truncation fix on Hopper at this budget**; the community claim of a large improvement is not supported here (but only one task, one budget).
- **LP-DS as released did not improve on Hopper within 200k raw steps.** Its evaluation curve is flat (1570-1620) and ends below its own first evaluation. With the shipped default delta (0.02 on Hopper; the paper table lists 0.35-0.50, unresolved) the policy barely moves from the base policy in this budget. This is **not** a test of whether LP-DS beats DSRL under matched tuning (the question H4 asked): neither method was tuned, the budget is ~1% of the LP-DS default, and arm E was not run.

### 2.3 Audit and engineering findings (`docs/audit.md`)
Confirmed in the pinned DSRL code (we read the lines): hard-coded `truncated=False` (time limit folded into `done`); `total_timesteps` counts chunk-steps and is hard-coded; DSRL-NA ignores `action_magnitude`; the double denoise exists only in the warm-up rollout. The released code writes eval metrics only to wandb, so we added a logging-only patch (0003) and a step-count stop rule (0002); neither changes learning. LP-DS differs from DSRL beyond the shift (deterministic actor, no entropy tuning, Polyak-averaged acting actor, per-dimension l2 normalization in the trust region), so LP-DS-vs-DSRL comparisons are confounded by more than the shift.

### 2.4 Cost per method (sub-question e)
| Item | Measured |
|---|---|
| Toy grid job | ~31.7 min per config (10-seed ensemble), RTX 3090 |
| DSRL-SAC Hopper, 200k raw steps | ~55-70 min/run when 5-10 run concurrently on an 18-32 vCPU pod; CPU-bound (simulation), the GPU is not the bottleneck |
| LP-DS Hopper, same budget | ~107-113 min/run at 4 concurrent on a 7.7-CPU pod (a different pod from DSRL's: **speed ratio is confounded by hardware**) |
| Laptop (GTX 1650, throttled CPU) | ~2.5 h per DSRL run: ~700 laptop-hours for the full original plan, which is why the plan moved to the cloud and was cut |
Per-run wall-clock is in `results/gym_stage*/runs/*/status.log`; the toy ledger is in `results/cloud_grid/`.

## 3. Failure modes: what the evidence says
| Failure mode | Evidence |
|---|---|
| Missing support | Not a failure in the dense-reward toy (H1(ii) not supported); untested with sparse reward |
| Noise bound / entropy target too tight | Supported in the toy (entropy-target sweep, 9/9 configs); not tested on Gym |
| Weak base | Probe predicts poorly at low weakness (negative rho at 0.05); entropy competitive at higher weakness (exploratory) |
| Mode collapse / reward hacking | **Not tested** (no diversity metric, no sparse-vs-dense or top-trajectory inspection) |
| Training budget too small to move the policy | LP-DS in Gym at 200k raw steps (observed, mechanism not isolated) |

## 4. NOT PROVEN
1. That the search-ceiling probe predicts steering headroom better than entropy (H3): the margin is 0.011 in rho with a [0.42, 0.91] CI, in 27 synthetic configs, with exploratory counter-evidence in subgroups.
2. Anything about real decoders from the toy results: the toy is a 2-D task with a trained flow decoder, not a robot policy.
3. That steering cannot help when the optimum is outside the base support (H1(ii) was contradicted in the dense-reward toy; sparse reward untested).
4. Whether LP-DS beats DSRL under matched tuning and budgets (H4), whether DSRL's clip/bound setting disadvantages the baseline (H6), or whether the entropy target matters on Gym (H7). None of arms D, E, F, G was run.
5. Anything on Walker2d, HalfCheetah, Robomimic Can/Square, or any non-Gym task. All Gym conclusions are Hopper, one checkpoint, 200k raw steps.
6. That the truncation fix is useless: only that no effect was detectable on Hopper at this budget (CI -72 to +56).
7. That LP-DS cannot learn: with default delta it did not move in this budget; longer training and its paper deltas were not tried.
8. Speed ratio between LP-DS and DSRL (different pods).
9. Agreement with the papers' headline numbers (not compared), the OpenReview review status (unreadable), the license status of DSRL (no LICENSE file; used for non-commercial research reading, cited).
10. Reward hacking, mode collapse and diversity-utility claims (not tested).

## 5. Threats to validity
- Hopper-only conclusions; 8-10 seeds; stage-1 metric was a single noisy evaluation (superseded by stage 2).
- LP-DS evaluation uses its own evaluation function (same 25 eval envs and 8 episodes per env, same reward averaging by inspection, not verified identical); its first-evaluation value (~1703) differs from DSRL's (~1605), consistent with evaluation noise or a different evaluation path.
- Pod hardware differed between stages; only wall-clock is affected, not learning (concurrency does not change learning, but stage-2 crashes show contention can kill runs; the 15 crashed runs wrote no data, so the retry was not selected on outcome; logged in `LOG.md`).
- Probe CIs are bootstrap over 27 configs, not independent tasks; configs share structure.
- Released checkpoints verified by checksum, not by provenance.

## 6. Reproducing
`README.md` lists the commands. Pre-registrations were committed before the runs they govern (git history); deviations are recorded as dated amendments or in `LOG.md`.
