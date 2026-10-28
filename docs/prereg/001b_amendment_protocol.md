# Amendment 001b to pre-registration 001 (2026-10-28)

Written BEFORE any tuning or reporting run for 001. It fixes things 001 left open. 001 and 001a are unchanged. No steering result has been seen at the time of writing (only a non-registered plumbing smoke test: inside/0.15/0.2, seeds 900-909, 5,000 env steps, which reached near-optimal return and is not used for any decision below).

## Fixed here
1. **Main-grid noise bound:** b_W = 1.5 for all 27 configs (inside DSRL's recommended [1,3] range; H2 varies it separately).
2. **Tuning protocol (the "3 trials per method" in 001):** learning rate in {1e-4, 3e-4, 1e-3}, tau fixed at 0.005, everything else as in 001. Tuned ONCE on config inside/0.15/0.2 with seeds 100-104 at a reduced budget of 8,000 env steps (UTD 20), selecting by mean dense return on a held-out tuning-state stream; the chosen learning rate is then used for all configs. Threat registered: tuning on an easy config may select poorly for hard ones; reduced-budget tuning may favour fast learners.
3. **Decoders:** one frozen decoder per config, trained with seed 0. Between-decoder variance is therefore NOT in the seed CIs (the 10 seeds vary the steerer only). Reported as a threat; an extra-decoder-seeds check is an exploratory follow-up.
4. **Evaluation / probe sizes:** J0 and C_N (N in {16, 64, 256, 1024}) on 2,000 held-out states; C_ref (N = 100,000) on 100 states; H_ent and variance on 2,000 states x 256 samples; steered return on 2,000 held-out states with deterministic actions; reachability gap per 001a. All evaluation streams are separate from training streams.
5. **Order of the first unattended session (laptop, time-boxed, chosen now, not from results):** tuning; then outside/0.15/0.2 (the empirically unreachable case, gap 0.81 in the decoder inspection); then a full-budget check of the steerer on the analytic bound-limited task (b_w = 0.8, target (1.5,-1.0), 50,000 env steps, 4 seeds) to close the open question of whether the gap to the analytic optimum (-0.728) closes at full budget. Remaining grid configs follow in later sessions in the order of `toy_grid()`.
6. **Compute log:** every run goes through `RunLedger`; laptop cost is recorded as $0 with hardware (CPU firmware-locked at 800 MHz) so wall-clock is reported honestly.

## Not changed
H1-H3 definitions, the primary test, the kill rule, N = 256, seeds 0-9 for reporting.
