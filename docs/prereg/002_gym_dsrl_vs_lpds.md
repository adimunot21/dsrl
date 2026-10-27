# Pre-registration 002: DSRL vs LP-DS under matched budgets (Gym)

Date written: 2026-10-27, before any run in this file. Not yet run: no DSRL or LP-DS training has been executed in this project.
Labels: arms A and B are **REPRODUCTION** attempts (released code, released checkpoints, our pins). Arms C-E and the bug-fix arms are **NEW EXTENSION**. Nothing here is a robot-policy result beyond the Gym tasks named.

## Question (sub-question b)
Do LP-DS's reported gains over DSRL hold when DSRL gets (i) its recommended per-dimension noise bound, (ii) the same number of tuning trials as LP-DS, and (iii) matched environment steps, UTD, network size and decoder calls per action?

## Facts from the audit that shape the design (see docs/audit.md)
- The released DSRL code has a hard-coded `truncated=False` (time limits folded into `done`), a warm-up double-denoise in SAC `collect_rollouts`, no `action_magnitude` use in DSRL-NA, and a hard-coded `total_timesteps` that counts chunk-steps. A community report says fixing `truncated` greatly helps SAC. These can confound any DSRL-vs-LP-DS gap, so the **as-released** and **bug-fixed** DSRL are separate arms.
- LP-DS differs from DSRL beyond the shift: deterministic actor with no entropy tuning, Polyak-averaged acting actor, different noise bound for HalfCheetah (1 vs 1.5), different network size for Transport. LP-DS's trust-region quantity is per-dimension-normalised in code. Its code deltas for Gym tasks (0.01-0.02) differ from the paper's reported table (unconfirmed).
- LP-DS ships no baseline configs and no stated delta-selection protocol.

## Arms (all on Hopper, Walker2d, HalfCheetah; released base-policy checkpoints)
| Arm | Description | Label |
|---|---|---|
| A | DSRL-SAC as released (our pinned commit, no code changes) | REPRODUCTION |
| B | LP-DS as released, paper/default delta from its config | REPRODUCTION |
| C | DSRL-SAC with only the minimal bug fixes (`truncated`, timestep accounting) applied as a patch file kept outside the submodule | NEW EXTENSION |
| D | Arm C with b_W tuned over {1, 1.5, 2.5} on tuning seeds | NEW EXTENSION |
| E | LP-DS with delta tuned over {0.01, 0.05, 0.1, 0.35} on tuning seeds (same number of trials as D) | NEW EXTENSION |
| F | DSRL with LP-DS's looser clip (||w|| <= 100) to test the claim that this setting disadvantages the baseline | NEW EXTENSION |

Bug-fix patches and any LP-DS edits are stored as diffs in `patches/` and applied to a clean checkout at run time. Nothing is copied into this repo.

## Matching rules
- Same environment steps (raw simulator steps, not chunk-steps), UTD, train frequency, network size per task, decoder denoising steps and calls per action, batch size, optimizer settings, replay size and warm-up. Anything that cannot be matched (LP-DS's actor type) is listed as a residual difference in the report, not hidden.
- Tuning: arms D and E each get **4 tuning trials**, with 3 tuning seeds per trial (seeds 100-102), selecting by mean return on those seeds. Arms A, B, C, F get no tuning. Tuning-seed data is never used in reported results.
- Reporting: seeds 0-9 per arm per task, evaluation on separate start-state RNG streams and evaluation-only episodes. If cost forces fewer than 10 seeds, the floor is 6 and the report states the reduced n; the choice is made from the cost gate below, not from results.

## Budget gate (before any run)
The raw env-step budget T per task is set **after** measuring throughput of the upstream pipeline on the target machine (laptop, then a short cloud trial), as: the largest T such that arms A-F x 10 seeds x 3 tasks plus tuning fits within the approved cap with 20% reserve. T is written into `configs/` and committed before the first reported run. If even T = 200k raw steps does not fit, the experiment is reduced to Hopper only and this is reported as a scope reduction.

## Hypotheses and decisions
- **H4.** The LP-DS-minus-DSRL gap shrinks when DSRL receives matched tuning and its recommended bound. Primary test, per task: paired bootstrap CI (over seeds, paired by seed index) of final return difference B - A (as released) versus E - D (matched tuning). H4 is supported on a task if the CI for E - D lies strictly closer to 0 than B - A's point estimate AND includes 0 or is under half of B - A in magnitude. Final return = mean over the last 10% of training evaluation checkpoints, fixed now.
- **H5 (bug confound).** Arm C outperforms arm A by more than the bootstrap CI half-width on at least one task. A negative result bounds how much of the paper-level SAC-vs-NA gap can be a code artefact.
- **H6 (clip).** Arm F is worse than arm D. If not, LP-DS's Fig. 2 baseline setting is not the source of its reported gap.
- Sample efficiency is a secondary metric: env steps to reach 80% of each arm's own final return, reported with bootstrap CIs; not used for decisions.
- One primary test per hypothesis per task. No multiple-comparison correction; the report presents three tasks as three separate, individually labelled cases.

## Reward hacking and utility
- Gym locomotion returns can be inflated by unusual gaits. Inspect the top-return episodes per arm (video or joint trajectory summary) and report any with non-standard gaits. Diversity claims need a downstream return metric: entropy of decoded actions is reported but never used alone.

## Costs recorded per run (sub-question e)
Wall-clock, GPU-hours, CPU core-hours, $ (from the ledger), env steps, gradient steps, decoder calls, peak memory. The report gives return per dollar and return per GPU-hour next to return per env step.

## Kill / stopping
- If arm A fails to reproduce released-code behaviour (return not above the base policy by any CI-detectable margin on any task) after the audited fixes in arm C, stop and write a failure analysis; this is a reproduction failure, not an LP-DS finding.
- Spend reaches 80% of the approved cap: stop and report.
- Any source claim contradicted by these runs is reported as contradicted, with the run IDs.

## Threats to validity registered in advance
- Released checkpoints may not match the paper's (Drive folder undocumented); we log checksums.
- Tuning grids are small by design; a larger grid for either method could change conclusions.
- The laptop CPU is firmware-throttled to 800 MHz, so timings from it are pessimistic. Cloud timings are the cost reference.
- LP-DS's repo is adapted from the unlicensed DSRL repo and bundles weights of unknown provenance; we use LP-DS's own base-policy weights only if they match the DSRL checkpoints by checksum, otherwise we use DSRL's.
