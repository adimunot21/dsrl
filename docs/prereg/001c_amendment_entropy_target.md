# Amendment 001c to pre-registration 001 (2026-10-28)

Written after session 1 and the entropy diagnostic (scripts/entropy_bound_test.py, results/entropy_bound_test_*.json), BEFORE any further 001 run. Session 1 results stand as run and are not re-selected.

## Finding (analytic bound task, b_w = 0.8, optimum -0.728, 4 seeds, 3,000 env steps; DIAGNOSTIC, not a registered test)
Deterministic return and gap to optimum by SAC target entropy: 0.0 -> -1.072 (gap -0.344, policy std 0.33); -2 (-dim) -> -0.834 (-0.106, std 0.15); -6 -> -0.741 (-0.013, std 0.025); -20 -> -0.728 (0.000). Max-entropy SAC with a tanh-Gaussian actor keeps the deterministic noise inside the box, by an amount set by the target entropy. The 50k-step full-budget gap (-0.104) matched the 1.6k-step gap (-0.107), so the earlier "under-training" explanation is refuted for this task.
Code fact: DSRL's released configs set `target_ent: 0.0` with `ent_coef: auto` (train_dsrl.py:98-100, cfg/*), i.e. a higher entropy target than the standard -dim.

## Consequences for 001
1. Session 1 (outside/0.15/0.2) used target entropy -dim (-2). It is NOT DSRL's setting. Its steerer return (-1.356) therefore reflects a steerer that under-uses the box by ~0.1, and the "C32-level" gap to noise-search ceilings is partly this effect. Not re-run yet.
2. The effective noise bound of a SAC steerer is smaller than b_W and depends on target entropy. H2 (gain vs b_W) is therefore confounded with the entropy target. H2 is amended to be evaluated at TWO entropy targets, 0.0 (DSRL) and -dim (standard), reported separately, never pooled.
3. The ceiling C_N uses unbounded N(0,I) noise while the steerer is box-bounded and entropy-regularised. A box-restricted ceiling C_N^box (the best of N noises drawn from the prior and clipped/rejected to the box |w| <= b_W) is added as a second probe reading. H3's primary test is unchanged (C_256, unrestricted); C_256^box is reported as a secondary predictor and is not used to rescue a failing H3.
4. Remaining main-grid configs run with target entropy 0.0 (matching the method under test, DSRL) AND -dim only on the 9-config subset support in {inside, edge, outside} x concentration in {0.05, 0.15, 0.4} at weakness 0.2. outside/0.15/0.2 must be re-run at 0.0 for consistency.

## Not changed
H1 and H3 definitions, the primary test and kill rule, seeds, N = 256.
