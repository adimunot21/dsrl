# Running log

## 2026-10-27
- Decisions (user): cloud cap ~$100, laptop-first; scope toy + Gym + Robomimic Can/Square; third-party code as pinned submodules, never vendored; open-ended timeline; use parallel agents.
- Phase 0 audit done; see docs/audit.md. No spend, no pushes, no downloads.
- Surprise: all four issue #6 claims hold in code, but the double denoising is limited to warm-up rollouts.
- Surprise: LP-DS's trust-region quantity is per-dimension-normalized in code, so its delta is not the paper's delta; deltas in the repo differ from the paper's table for Gym tasks (unconfirmed).
- Process note: subagent reports are unverified until I check the cited lines. 7 of the claims above I re-checked in code.
- User decisions (2026-10-27): skip the license query to the DSRL authors; public Drive checkpoints may be used (non-commercial, cite the papers). We still keep third-party code as pinned submodules, not copied into our tree.
- Toy decoder inspection: "outside support" at mode std 0.4 is tail-reachable (min dist 0.015 over 100k noises), and decoder blur widens modes. Recorded as pre-registration amendment 001a before any steering run; 001 untouched.
- GPU decoder training is not bitwise reproducible; seeds fix data and init but not CUDA kernels. Document as a threat to validity.
- Upstream DSRL env builds (conda env dsrl_upstream, py3.9) and steps Can/Hopper via the repo wrappers. Raw random-action throughput (machine loaded, pessimistic): Hopper 156-268 chunk-steps/s at n_envs=4; Can 7-9 chunk-steps/s at n_envs=4 (robosuite OSC controller in Python dominates, ~93 ms/sim step). 100k Can chunk-steps is ~3-4 h before any gradient cost. Used a dummy normalization.npz; real one is in the Drive checkpoints (not downloaded).
- Steerer (own SAC) passes 4 known-answer tests. The agent loosened the bound-limited tolerance 0.1 -> 0.15 and set utd 4 after a first failure (reward -0.842 vs analytic optimum -0.728 at 2000 steps); it never exceeded the bound. NOT yet verified that the gap closes at the full 50k-step budget; do that before using the b_W sweep.
- Compute blocker: steerer costs ~44-58 ms per gradient step on this machine (CPU 800 MHz under load, governor powersave, no_turbo=1; GPU path is launch-overhead-bound, 43 ms). Prereg 001 needs 1M gradient steps/run => ~14 h/run. Infeasible for 27 configs x 10 seeds. Needs a fix before any prereg run (governor change needs sudo; and/or batched multi-seed training).
- Wrote prereg 002 (Gym DSRL vs LP-DS) before any Gym run. Budget T deferred to a throughput gate and committed in configs/ before first reported run.
- Batched ensemble steerer (S seeds as one stacked network, CUDA graph): 8 tests pass (re-run by me). Measured S=10 ensemble step 11.7 ms on the GTX 1650 (1.2 ms per member-step), vs ~44-65 ms unbatched.
- Smoke test (scripts/smoke_toy_ensemble.py, seeds 900-909, NOT a pre-registered run, not ledgered): inside/0.15/0.2, 5000 env steps; J0 -1.343, C256 -0.022, H_probe 1.321, steered return ~-0.002 on all 10 seeds (G 1.340). Plumbing only; easy config.
- Compute: prereg 001 (1M grad steps, 10 seeds) = ~3.3 h per config on the laptop; 27 configs ~88 h; full H2 sweep on all configs would add ~470 h. Needs cloud GPU or a pre-run amendment cutting the H2 sweep. Decision pending with user.
- Laptop CPU is firmware-locked at 800 MHz (governor/platform_profile set to performance, no_turbo write refused). A second-session prompt was given to the user to investigate (charger/BD PROCHOT suspected, unverified).
- Runner bug found by plumbing test: steerer_batched finiteness check fired before the first update (last=NaN); fixed. Session 1 launched 2026-10-28 via scripts/overnight_001.py (tune 3 lr -> outside/0.15/0.2 full budget -> bound check).
- Downloaded only the Gym subset of the public Drive folder (3 normalization.npz + 3 pretrain state_3000.pt, ~4.5 MB each) to data/dppo_log (gitignored); sha256 in docs/checkpoint_manifest.json. Checkpoints are torch pickles: load only with weights_only=True in the isolated dsrl_upstream env. Hopper/Walker/HalfCheetah medium-v2 ta4_td20; Robomimic checkpoints not downloaded.
- SESSION 1 RESULTS (2026-10-28, laptop, ledgered, results/session1/):
  * Tuning (inside/0.15/0.2, seeds 100-104, 8k env steps): lr 1e-4/3e-4/1e-3 gave mean return -0.0029/-0.0025/-0.0026: a statistical tie; lr=3e-4 selected by protocol. Registered threat realised: tuning on an easy config does not discriminate.
  * outside/0.15/0.2 (10 seeds, 50k env steps, 3.02 GPU-h): J0 -2.688, steered -1.356 on every seed (spread 0.002), C16 -1.423 C256 -1.176 C1024 -1.095 C_ref -0.885; H_probe(256)=1.512, gain 1.332; plateau by ~30k env steps. Gain < H_probe and < C_ref headroom (consistent with H1(i) on this config only).
  * Bound check (b_w=0.8, 50k steps, 4 seeds, 1.27 GPU-h): returns -0.832 vs analytic optimum -0.728; gap -0.104, NOT closed vs -0.107 at 1.6k steps. The agent's "under-training" explanation is refuted; cause unverified (suspect SAC entropy regularisation keeping the tanh policy inside the bound; alpha stays ~0.0185). Never exceeded the bound.
- Entropy diagnostic (3,000 env steps, 4 seeds, analytic task): gap to optimum 0.0->-0.344, -2->-0.106, -6->-0.013, -20->0.000: SAC entropy regularisation (not under-training) explains the bound gap. DSRL's released configs use target_ent 0.0, verified train_dsrl.py:98-100. Session 1 used -dim, so it is NOT DSRL's setting; amendments 001c and 002a record consequences and add arm G (entropy-target control).

## 2026-10-28 Cloud benchmark (RunPod community, RTX 3090 24 GB, AMD EPYC 7H12 shared 256-thread host)
- Run: toy outside/0.15/0.2, 10 seeds, 50k env steps, target entropy 0.0 (DSRL's setting), OMP_NUM_THREADS=4. Commit 26e55f0.
- Wall-clock 1884 s (31.4 min) for the whole run incl. decoder, probes (~40 s) and training (~1650 s). Laptop session 1: ~3.1 h per config => ~6x faster.
- Probes identical to the laptop run (J0 -2.6878, C256 -1.1755, H_probe 1.512, gap 0.809): cross-hardware determinism of the probe path.
- Steered return -1.411 (deterministic eval), gain 1.276 (seed spread 1.275-1.277), vs 1.332 at target -dim in session 1. Lower gain at target 0.0, as the entropy diagnostic predicted. Single config, so not yet a test of any hypothesis.
- Pitfall: torch on the 256-thread host without a thread cap oversubscribes (3600% CPU); the test suite took 18m40s. Always cap OMP_NUM_THREADS.
- $ not yet recorded: hourly rate to be read off the RunPod billing page by the user. Pod must be stopped by the user.
- Access: RunPod proxy SSH needs commands piped to an interactive shell (no direct exec).

## 2026-10-29 Cloud spend (user-reported)
- User reports ~$10 spent in total since the pod was started (RunPod community RTX 3090, $0.5/h): setup + tests + benchmark + 36-job grid. Approximate, from the user's memory of the billing page; replace with the exact figure if given.
- Cumulative project cloud spend: ~$10 of the $100 cap (stop line $80). 36 jobs x ~31.7 min = ~19 h GPU for the grid.

## 2026-10-29 Pre-registered grid analysis (scripts/analyze_toy_grid.py -> results/toy_grid_analysis.json)
- H3 primary: rho(H_probe_256, G)=0.736, CI [0.420, 0.912]; rho(H_ent, G)=0.409. By the letter SUPPORTED (rho>=0.7 and CI lower bound 0.420 > 0.409), but the margin over entropy is 0.011 -> fragile. Kill rule (rho<0.5) not triggered.
- Secondary (not used to support H3): box ceiling rho=0.944 [0.822,0.990]; action variance 0.499; C_ref headroom 0.653.
- H1(i): 0/270 (config,seed) runs exceed C_ref-J0+0.05 -> passes.
- H1(ii): exactly 6 unreachable configs (all outside, conc 0.05/0.15); mean gain 1.163 CI [0.979,1.330] vs inside 1.248 -> NOT SUPPORTED. The pre-registered expectation (gain ~0 when outside) was wrong under dense reward: partial credit for getting closer. Not redefined post hoc.
- Entropy subset (9 configs, exploratory): target -2 gives higher gain than 0.0 in 9/9 configs (+0.01 to +0.09).

## 2026-10-29 Gym phase: laptop throughput gate + patch 0001
- Arm A (algorithm=dsrl_sac; NOTE the released Gym config default is dsrl_na) on Hopper, tuning seed 100, 15 min: 5.65 chunk-timesteps/s = ~22.6 raw sim steps/s => ~2.5 h per 200k-raw-step run. Base policy ep_rew_mean ~1470 at start. Base policy loaded with weights_only=True (verified in dppo/model/diffusion/diffusion_eval.py:33-35).
- Full arm grid ~270 runs => ~700 laptop-hours: laptop route infeasible beyond a few runs. Need a cloud rate gate before fixing T (prereg 002: T in configs/ before first reported run).
- patches/0001_truncation_and_raw_step_budget.patch (applies cleanly to dsrl 8d21b9c): (1) ActionChunkWrapper reports terminated/truncated separately using the inner env's TimeLimit.truncated info and the chunk wrapper's own limit, so time-outs are bootstrapped; (2) total_timesteps = cfg.total_raw_env_steps // act_steps (default 80M raw == released 20M chunk-steps). Not yet run; arm C = A + this patch.
- Launch recipe (no submodule edits): cd third_party/dsrl; LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia MUJOCO_GL=egl python train_dsrl.py --config-path $PWD/cfg/gym --config-name dsrl_hopper.yaml algorithm=dsrl_sac use_wandb=False seed=S log_dir=... base_policy_path=<abs> normalization_path=<abs>

## 2026-10-30 Gym stage 1 result (Hopper, 200k raw steps, seeds 0-9; pod RTX 3090 @ $0.507/h, auto-stopped after verification)
- All 20 runs finished (~68 min each, 10 concurrent, 2 waves), 0 failures, pod stopped by watchdog path (SSH now "container not found").
- Metric (prereg 002 / gym_budget.json): mean eval reward over last 10% of eval checkpoints. Eval interval was 16,000 chunk-steps, so each run has only 4 evals (0, 16k, 32k, 48k) -> the "last 10%" is the single final eval. This is noisy; flagged as a limitation.
- Arm A (released DSRL-SAC): 1908 [1826, 2002] (bootstrap CI over seeds); base policy initial eval ~1616-1631. Gain over initial +292 [211, 379].
- Arm C (truncation fix): 1763 [1702, 1833]; gain +133 [59, 212].
- C - A paired by seed: -144 [-248, -38]; C > A in 2/10 seeds. The truncation fix did NOT help here; it lowered the final return. Not the direction H5 anticipated; reported as is, no redefinition. Mechanism not established (possible: bootstrapping through time-limit changes value targets/exploration; or eval noise with 4 evals). Total ~ $2.3-2.8 spent of ~$10.

## 2026-10-30 Stage 2 launch incident (A40-class replacement: RTX 4000 Ada pod, 18 vCPU, 71 GB, ~$0.5/h)
- 10 concurrent runs x 29 env workers on 18 vCPUs: 15 of 20 runs crashed at startup (BrokenPipe/ConnectionReset in SubprocVecEnv workers) within ~20 min; the 5 that started later ran fine (3380-4180 s). Cause not confirmed (CPU/pid/shm contention suspected). Crashed runs wrote no eval.jsonl -> no outcome-dependent selection. Re-running the 15 with concurrency 5 (scripts/run_gym_stage2_retry.sh); crashed logs kept as *_crashed1.

## 2026-10-30 Gym stage 2 result (dense-eval replicate, Hopper, seeds 10-19, prereg 002b; pod stopped by verified auto-stop at ~15:32 UTC)
- All 20 runs completed (13 evals each); 15 of them via the concurrency-5 retry after the startup crashes. Pod: RTX 4000 Ada, 18 vCPU, ~$0.5/h, ~4.7 h total incl. setup -> ~$2.4. Cumulative cloud spend this session ~$5 of the ~$10 (user also authorised up to $7-8 more).
- Primary metric (mean of last 3 evals): A 1808 [1758, 1868]; C 1804 [1753, 1856]; C - A = -4.6 [-72, +56]; C > A in 7/10 seeds. Verdict per 002b: NO DETECTABLE DIFFERENCE. The stage-1 deficit of C (-144 [-248,-38]) did not replicate on independent seeds with denser eval.
- Secondary (mean over all evals after step 0): C - A = -25.7 [-68, +13]; not significant.
- Pooled 20 seeds (mixed metrics: stage-1 single final eval + stage-2 last-3): -74.6 [-148, -8]; reported but not a primary result (metric mismatch; stage-1 effect likely partly noise/winner's-curse).
- Interpretation: truncation fix has no demonstrated effect on Hopper at 200k raw steps. A also gains over base policy (~1610) in both stages: +200 here.

## 2026-10-30/31 Gym stage 3 result (arm B = LP-DS as released vs arm A, Hopper, prereg 002c)
- Pod: A40, ~7.7 effective CPUs (cgroup cpu.max 765000/100000), 4 concurrent. LP-DS runs took 6400-6800 s each (~110 min), ~1.8x slower than DSRL arm A on the 18-vCPU pod (different pod; confound). My estimate (~60 min/run) was wrong and I did not recheck speed after the first sync; three waves did not fit the 300-min hard limit (stopped ~22:13 UTC). Seeds 18, 19 were cut off at 36,000/48,000 chunk-steps (logs synced; excluded from the paired test, floor of 6 seeds in prereg 002).
- Paired on seeds 10-17 (n=8), metric = mean of last 3 evals: A 1824 [1763, 1894]; B 1591 [1584, 1598]; B - A = -233 [-302, -173]; B > A in 0/8 seeds. Verdict per 002c: A (DSRL) better.
- B's evals are flat (1570-1620 throughout; per-run gain over its own first eval about -112, while its first eval was 1703 vs A's 1605: different eval functions/noise), i.e. with the released default delta (0.02) and a 200k-raw-step budget, LP-DS did not learn measurably on Hopper. Caveats: the budget is ~1% of the repo defaults (20M timesteps); the paper's delta table (0.35-0.50) is unresolved; LP-DS was not tuned (arm E not run), so this does not test "LP-DS beats DSRL under matched tuning". Pod/CPU differences affect wall-clock only, not learning.
- Cost: ~5 h pod time (setup 5 min + 4 h 55 m) at an unconfirmed A40 rate (assume <= $0.5/h => <= $2.5). Cumulative cloud spend this session ~ $7.5-8 (user said they had ~$10 left at the start of the Gym phase and authorised 7-8 more).
