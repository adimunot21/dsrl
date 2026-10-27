# Phase 0 audit (2026-10-27)

Evidence levels: **CODE** = I read the cited line myself in a pinned clone. **AGENT** = reported by a subagent, not re-checked by me. **SUMMARIZED** = from a web page read through a summarizing model, so quotes are secondhand. Nothing here is a result; reading is not a result.

## Pins
| Component | Commit | Notes |
|---|---|---|
| ajwagen/dsrl | 8d21b9cf5545 | submodules not checked out |
| ajwagen/dppo-dsrl | 86ce51834055 | gitlink only |
| ajwagen/stable-baselines3-dsrl | 10e5d311b5c4 | gitlink only; `.gitmodules` uses SSH URLs |
| HikmetSimsir/lpds | 6ac778a5ccce | |

## Hardware
i7-9750H (12 threads), 31 GB RAM, GTX 1650 4 GB, driver 595.99.02, 713 GB free disk. Python 3.9 / conda locks not yet created.

## Issue #6 claims
| Claim | Verdict | Evidence |
|---|---|---|
| SAC path denoises twice | CONFIRMED, narrower than stated: `collect_rollouts` (warm-up) only | CODE `utils.py:171`, `env_utils.py:216-217`. That `model.learn()` denoises once is AGENT only (submodule file not read). Effect size on results is unmeasured. |
| NA ignores `action_magnitude` | CONFIRMED | CODE: all uses are SAC-only (`utils.py:169-170`, `env_utils.py:199-200`) |
| `truncated` hard-coded False | CONFIRMED | CODE `env_utils.py:184`; time limit folded into `done` at :180 |
| `total_timesteps` not scaled by chunk | CONFIRMED | CODE `train_dsrl.py:178` (20,000,000, hard-coded); chunk factor only in logging, `utils.py:77` |

Community claim, unverified by us: setting terminated/truncated correctly gives a "huge" improvement for SAC and little NA-over-SAC gain afterward (issue #6, omkarpatil18). This bears on the paper's "SAC needs ~2x samples" result and must be tested, not assumed.
Other CODE facts: `wandb`, `gym`, `d4rl` imported unconditionally (`train_dsrl.py:7,11,12`, `utils.py:2`). `action_magnitude` is 1.5 for can/lift/square/hopper/halfcheetah, 1.0 transport, 2.5 walker (AGENT).

## Issues #2, #5, paper claims (AGENT / SUMMARIZED)
- #2: reporter says ~100 h to reproduce Can; `n_envs=50` "works much better". No maintainer reply. Single anecdote, GPU not stated.
- #5: Transport not matching after 4 days on an RTX 5090. No reply, 0 comments.
- Paper claims confirmed from arXiv HTML (SUMMARIZED): SAC ~2x samples vs NA (Sec 5.6); b_W in [1,3] online (App. B); UTD >= 20; layer norm; 5 seeds online, error bars 1 s.e. (App. C.1), Table 1 "±" is 1 s.d. (App. C.2). No wall-clock or GPU-hour numbers reported.
- PMLR page unreachable; OpenReview reviews unreadable for both papers. Review status remains UNVERIFIED.

## LP-DS (AGENT unless marked)
- Shift: v1 is state-only, v2 is state+noise; released code is state+noise (matches v2). Sources: arXiv HTML, SUMMARIZED; code `lpds.py:52-80, :532`, CODE for 52-60 and 530-534.
- Trust region in code differs from the paper: l2 divided by the number of active dimensions (CODE `lpds.py:436`), plus EMA, error clipping and log-parameterized lambda (AGENT). So the paper's delta and the code's delta are not the same quantity.
- Code deltas (CODE, configs): Hopper 0.02, Walker 0.01, HalfCheetah 0.01, Square/Can 0.35, Lift 0.05, Transport 0.35, Pen/Door/Relocate 0.35, Hammer 0.1. The agent reports the paper's Table 2 lists 0.50/0.35/0.35 for the Gym envs; that comparison is SUMMARIZED and unconfirmed.
- No stated delta-selection protocol or tuning split. Delta is "selected to balance performance and behavioral diversity" (SUMMARIZED).
- Fig. 2 caption: DSRL under "the same hard clip (||w|| <= 100)" (SUMMARIZED). The repo contains no baseline configs and its own code clamps per dimension to +-b_W, so the baseline setting cannot be reproduced from the repo.
- Seeds: 6 main, 3 for Fig. 2/5 and ablations (SUMMARIZED); every config sets `seed: 1`.
- Differences beyond the shift: deterministic actor with no entropy tuning, Polyak-averaged acting actor, HalfCheetah b_W 1 vs DSRL's 1.5, Transport net 1024 vs DSRL's 128. These confound any LP-DS-vs-DSRL gap and must be matched in Phase 4.
- Not done: full diff of `train_lpds.py` vs `train_dsrl.py`; critic-backup decoder-call comparison (DSRL SB3 fork was not checked out).

## Licenses (AGENT, except where stated)
- ajwagen/dsrl: NO LICENSE (CODE: none in clone; agent: API `license: null`). Reference only; never copy into our tree.
- dppo-dsrl, SB3 fork: MIT. LP-DS: MIT, but adapted from the unlicensed DSRL and bundles ~114 MB of `.pth` weights of unstated provenance.
- LP-DS bundles a package named `stable_baselines3` that collides with the DSRL fork: use separate environments.
- Drive checkpoint folder (public, four folders: gym, gym-pretrain, robomimic, robomimic-pretrain, no sizes): no license.

## Cloud (AGENT, SUMMARIZED, unverified)
Spot CPU is likely the best steps per dollar (e.g. 32 vCPU at ~$0.69/h); RunPod 4090/L4 ~$0.34-0.44/h; AWS Spot quota for new accounts is the main risk. Vast.ai and GCP unchecked. Re-price before any spend; benchmark throughput first.

## Open decisions for the user
1. Send a license query to the DSRL authors? (not sent)
2. Download the Drive checkpoints despite no license? (not downloaded)
3. Any cloud account creation / quota requests (none made).
