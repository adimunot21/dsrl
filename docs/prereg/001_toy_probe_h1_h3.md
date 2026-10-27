# Pre-registration 001: toy decoders, noise-search probe, H1-H3

Date written: 2026-10-27, before any experiment in this file has been run.
Label: **SIMPLIFIED MECHANISM TEST**. A toy decoder is not a robot-policy reproduction; no conclusion here transfers to Robomimic/Gym without Phase 4/5 evidence.

## Question
Does a forward-pass-only estimate of the best reward reachable through a frozen decoder (the *search ceiling*) predict how much DSRL-style noise steering improves the policy, better than the entropy of the base action distribution?

## Setup (fixed before running)
- **Task.** Contextual one-step task: state s in [-1,1]^2, action a in R^2, deterministic reward r(s,a) = -||a - g(s)||, with g(s) the goal for that state. Sparse variant: r = 1 if ||a - g(s)|| < 0.1 else 0 (for the reward-hacking comparison).
- **Frozen decoder.** A small conditional flow-matching model, trained once per toy config on data from a K-mode mixture, decoded deterministically with a fixed number of Euler steps from noise w in R^2. The decoder is never updated after training. Its parameters, training data seed and config hash are logged.
- **Toy factors (full grid, 3 x 3 x 3 = 27 configs):**
  - *Support:* goal inside the data support / on the edge / outside it (distance of g(s) from the nearest data mode: 0, 0.5, 1.5 data std).
  - *Concentration:* mode std 0.05 / 0.15 / 0.4.
  - *Weakness:* fraction of mixture mass on the best mode 0.5 / 0.2 / 0.05.
- **Steerer.** DSRL-SAC-style: a SAC agent whose action is the noise w, with box bound b_W; the decoder is part of the environment. Our own implementation in `dsrlstudy/`; upstream DSRL code is not copied (no license).
- **Budget match.** Every steerer run uses the same number of environment steps (50,000), gradient steps (UTD 20), decoder calls per action (1), and network size (3 x 128, layer norm).

## Quantities
- **Base return** J0: mean return of the decoder with w ~ N(0, I), 2,000 held-out start states.
- **Search ceiling** C_N: for each held-out state, draw N noises, decode, take the max reward, average over states. Reported for N in {16, 64, 256, 1024}. Reference ceiling C_ref uses N = 100,000.
- **Probe headroom** H_probe = C_256 - J0 (primary; N = 256 fixed here, before looking at results).
- **Achieved gain** G = J_steered - J0 where J_steered is the final return on held-out start states.
- **Entropy predictor** H_ent: kNN entropy estimate (k = 5) of base actions at each state, averaged over states. Secondary baseline predictor: variance of base actions.
- **Seeds.** Steerer hyperparameters (learning rate, tau) tuned on seeds 100-104 for 3 trials per method only; reported results use seeds 0-9 and start states drawn from a separate RNG stream. No selection on the reported seeds.

## Hypotheses and decisions
- **H1.** G is at most H_probe up to estimation noise, and G is near 0 when the support condition is "outside". Test: (i) the fraction of (config, seed) runs with G > C_ref - J0 + 0.05 (the ceiling is exceeded) should be under 5%; (ii) mean G in the "outside" configs, with a bootstrap CI over seeds, should include 0 or be below 10% of the mean G in "inside" configs.
- **H2.** G first rises and then plateaus or falls as b_W grows beyond the prior's range. Test: for each config, sweep b_W in {0.5, 1, 1.5, 2, 3, 5, 10}. The pre-specified criterion is that G at b_W = 10 is lower than the maximum G over the sweep by more than the bootstrap CI half-width, in at least 1/3 of configs. Otherwise H2 is not supported in this toy.
- **H3.** Primary test: Spearman correlation across the 27 configs (per-config mean over seeds) between H_probe and G, versus between H_ent and G. H3 is supported only if rho(H_probe, G) >= 0.7 and its bootstrap 95% CI lower bound exceeds rho(H_ent, G) point estimate. This is the single primary test; everything else is exploratory.

## Kill / stopping criteria
- If rho(H_probe, G) < 0.5 on the full grid, **stop and report the probe as non-predictive** in this toy, as the project's stopping rule requires. No re-definition of the probe after seeing results; any variant is a new, labelled, exploratory analysis.
- If SAC fails to improve on J0 in "inside" configs (G <= 0 for more than half of them), treat it as an implementation failure first: debug against a known case before drawing any conclusion about steering.

## Failure-mode checks (descriptive, not tested hypotheses)
- *Missing support:* H1 test above.
- *Noise bound too large:* H2 sweep.
- *Weak base policy:* weakness factor.
- *Mode collapse:* number of distinct decoded modes visited by the learned noise policy (clustered, distinct to 0.1) versus the number of rewarding modes; downstream utility is return, not entropy.
- *Reward hacking:* compare steerers trained on dense vs sparse reward on the same configs; inspect the top-1% return trajectories (here, actions) for decoder-exploit artefacts such as actions outside the data hull.

## Analysis plan
- Per-seed results always reported. Returns: bootstrap CI over seeds (10,000 resamples, seeds resampled). Success rates in the sparse variant: Wilson intervals. Correlations: Spearman with bootstrap CI over configs, resampling configs.
- 27 configs are not 27 independent tests; no per-config significance claims. Only H1(i)(ii), H2 and H3 as defined above.

## Threats to validity registered in advance
- Probe is **not purely forward passes** in general: it needs a reward evaluation for every candidate noise. In this one-step toy that is free; in multi-step environments it needs open-loop rollouts or a learned value function. Any claim about a "cheap probe" must report that cost.
- The toy is low-dimensional (2-D noise); noise search is much more effective here than in high-dimensional action chunks. Passing H3 here says little about real decoders.
- The decoder is trained by us on a synthetic mixture; its smoothness may differ from robot policies.
- Our SAC is a reimplementation, not the released code, so it can differ in ways the paper's results do not capture.
