# Amendment 002a to pre-registration 002 (2026-10-28)

Written before any Gym run. Motivation: 001c. DSRL's SAC uses target entropy 0.0 (verified in train_dsrl.py:98-100, cfg/gym/*.yaml); LP-DS uses a deterministic actor with no entropy tuning (audit, lpds.py / configs). A toy diagnostic shows target entropy alone moves a SAC steerer's effective noise bound by 0.1-0.35 on a 0.8 box. So part of any LP-DS-minus-DSRL gap could be an entropy effect, not a trust-region effect.
## Added
- **Arm G (NEW EXTENSION):** arm D (bug-fixed DSRL, tuned b_W) with target entropy -dim instead of 0.0, same tuning trials as D, to isolate the entropy-target contribution. Falls under the same budget gate and kill rules.
- **H7:** G differs from D by more than the paired-bootstrap CI half-width on at least one task. If so, the report must attribute the corresponding share of the LP-DS-vs-DSRL gap to entropy settings and not to the trust region.
## Not changed
Arms A-F, H4-H6, matching rules, budget gate.
