# DSRL steering study

When can a frozen, noise-controllable generative policy be improved by learning *which initial noise to feed it* (DSRL, Diffusion Steering via RL), and when does that fail? A pre-registered, simulation-only study with toy decoders, a cheap predictive probe, and a matched-budget Gym comparison of DSRL against LP-DS (arXiv 2606.01151).

**Read `docs/REPORT.md` for the results, scope cuts and the explicit NOT PROVEN list.**

## Headline results (see the report for caveats)
- **Toy decoders (SIMPLIFIED MECHANISM TEST):** a forward-pass-only noise-search ceiling predicted steering gain with Spearman 0.74 [0.42, 0.91] vs 0.41 for action entropy: passes the pre-registered rule narrowly; weak evidence. The "no gain when the optimum is outside the support" prediction failed under dense reward. The SAC entropy target acts as an effective noise bound (9/9 configs improved at -2 vs 0.0).
- **Gym Hopper (REPRODUCTION):** DSRL-SAC as released beats the frozen base policy in 20/20 seeds (+200 to +290 return). The truncation fix: no detectable effect (-5 [-72, +56], 10 fresh seeds). LP-DS as released (default delta) did not improve within 200k raw steps and ended 233 [173, 302] below DSRL (8 seeds), which is *not* a test of tuned LP-DS vs tuned DSRL.
- Budget was cut from ~$100 to ~$17 in total; Walker2d, HalfCheetah, Robomimic and the tuned arms were not run (see report).

## Layout
| Path | Contents |
|---|---|
| `docs/REPORT.md` | Final report |
| `docs/prereg/` | Pre-registrations and dated amendments (written before the runs they govern) |
| `docs/audit.md` | Code/claim audit of the DSRL and LP-DS releases, with evidence levels |
| `LOG.md` | Running decision log, incidents, per-stage results |
| `dsrlstudy/` | Our package: `NoisePolicy` interface, toy task/decoder, search-ceiling probe, batched SAC steerer, stats, run ledger |
| `scripts/` | Toy grid orchestrator and analysis, Gym stage runners, pod bootstrap, watchdog, analysis scripts |
| `patches/` | Diffs applied to clean copies of third-party code at run time (never vendored) |
| `configs/` | Pre-committed Gym budgets |
| `results/` | Raw eval logs, per-run results, analysis JSON |
| `third_party/` | `ajwagen/dsrl` and `HikmetSimsir/lpds` as pinned git submodules |
| `tests/`, `envs/` | 51 unit tests; locked environments |

## Reproduce
```bash
git clone --recurse-submodules https://github.com/adimunot21/dsrl && cd dsrl
conda create -n dsrlstudy --file envs/dsrlstudy.explicit.txt && conda activate dsrlstudy
PYTHONPATH=. pytest -q tests                       # 51 tests (~12-19 min)
python scripts/analyze_toy_grid.py                 # reproduces results/toy_grid_analysis.json
python scripts/analyze_gym_stage1.py; python scripts/analyze_gym_stage2.py; python scripts/analyze_gym_stage3.py
```
Re-running the training needs a GPU pod and the upstream Python 3.9 environment: `scripts/cloud_bootstrap.sh`, `scripts/cloud_bootstrap_upstream.sh`, `scripts/cloud_bootstrap_lpds_code.sh`, then `scripts/run_gym_stage{1,2,3}.sh` (and `run_gym_stage2_retry.sh`); `scripts/pod_watchdog.sh` stops a RunPod pod on a time limit. The Hopper base-policy checkpoint comes from the DSRL release (checksums in `docs/checkpoint_manifest.json`); checkpoints are untrusted pickles and are loaded with `weights_only=True`.

## Attribution and licenses
DSRL (Wagenmaker et al., arXiv 2506.15799) and LP-DS are third-party research code, referenced as submodules and patched at run time; the DSRL repository carries no LICENSE file, so it was used for non-commercial research and is cited here, not redistributed. Paper details cited from memory are marked as such in the docs.
