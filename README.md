# dsrl

A pre-registered, simulation-only study of DSRL (Diffusion Steering via Reinforcement Learning). DSRL improves a frozen generative policy by learning which initial noise to feed it. The study asks when that works and when it fails.

It has three parts: toy decoders to test the mechanism, a cheap probe that predicts how much steering will help, and a Gym comparison of DSRL against LP-DS (arXiv 2606.01151) at matched budget.

The full results, scope cuts and a list of what the study does not prove are in `docs/REPORT.md`.

## Results

- Toy decoders: the best score found by searching noise with forward passes only predicted the steering gain with Spearman 0.74 (CI 0.42 to 0.91), against 0.41 for action entropy. This passes the pre-registered rule narrowly, so the evidence is weak. The prediction that steering gives no gain when the optimum lies outside the decoder's support failed under dense reward. With the SAC entropy target set to -2 rather than 0.0, 9 of 9 configurations improved, so the target acts as an effective bound on the noise.
- Gym Hopper reproduction: DSRL-SAC as released beats the frozen base policy in 20 of 20 seeds, by +200 to +290 return. A fix for episode truncation had no detectable effect (-5, CI -72 to +56, 10 fresh seeds). LP-DS with its default settings did not improve within 200k steps and ended 233 below DSRL (CI 173 to 302, 8 seeds). That comparison involves untuned LP-DS only.
- The budget was cut from about $100 to about $17. Walker2d, HalfCheetah, Robomimic and the tuned arms were not run.

## Layout

| Path | Contents |
|---|---|
| `docs/REPORT.md` | Final report |
| `docs/prereg/` | Pre-registrations and dated amendments, written before the runs they cover |
| `docs/audit.md` | Audit of the DSRL and LP-DS releases, with evidence levels |
| `LOG.md` | Decision log, incidents and per-stage results |
| `dsrlstudy/` | Python package: noise-policy interface, toy task and decoder, search probe, batched SAC steerer, statistics, run ledger |
| `scripts/` | Experiment runners, analysis, cloud bootstrap and watchdog scripts |
| `patches/` | Diffs applied to clean copies of third-party code at run time |
| `configs/` | Gym budgets, fixed before the runs |
| `results/` | Raw logs, per-run results, analysis JSON |
| `third_party/` | `ajwagen/dsrl` and `HikmetSimsir/lpds` as pinned submodules |
| `tests/`, `envs/` | 51 unit tests and locked environments |

## Reproducing

    git clone --recurse-submodules https://github.com/adimunot21/dsrl && cd dsrl
    conda create -n dsrlstudy --file envs/dsrlstudy.explicit.txt && conda activate dsrlstudy
    PYTHONPATH=. pytest -q tests
    python scripts/analyze_toy_grid.py
    python scripts/analyze_gym_stage1.py; python scripts/analyze_gym_stage2.py; python scripts/analyze_gym_stage3.py

The tests take 12 to 19 minutes. Re-running the training needs a GPU pod and the upstream Python 3.9 environment. The scripts are `scripts/cloud_bootstrap*.sh` followed by `scripts/run_gym_stage{1,2,3}.sh`, and `scripts/pod_watchdog.sh` stops a RunPod pod at a time limit. The Hopper base-policy checkpoint comes from the DSRL release (checksums in `docs/checkpoint_manifest.json`) and is loaded with `weights_only=True`.

## Attribution

DSRL (Wagenmaker et al., arXiv 2506.15799) and LP-DS are third-party research code, referenced as submodules and patched at run time. The DSRL repository has no licence file, so it was used for non-commercial research and is cited here without being redistributed. Paper details cited from memory are marked as such in the docs.
