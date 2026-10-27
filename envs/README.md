# Environments

| Env | Files | Status |
|---|---|---|
| dsrlstudy (ours: analysis, toy experiments) | dsrlstudy.yml (intent), dsrlstudy.lock.yml (exact versions, no builds), dsrlstudy.explicit.txt (exact URLs, reproducible) | BUILT and verified |
| dsrl_upstream (third_party/dsrl) | dsrl_upstream.yml, .md, .lock.txt, dsrl_upstream_bench.py | BUILT, smoke-tested (Can + Hopper env stepping); no training, no checkpoints yet. See docs/upstream_build_log.md |
| lpds_upstream (third_party/lpds) | lpds_upstream.yml, lpds_upstream.md | SPEC ONLY, not built |

## Create
- dsrlstudy: `conda env create -f envs/dsrlstudy.yml` (solver picks newest), or exactly reproducible:
  `conda create -n dsrlstudy --file envs/dsrlstudy.explicit.txt`.
- Upstream envs: read the .md first; the .yml is only the conda/pip skeleton and has manual steps (MuJoCo 2.1 binaries, submodule checkout).
- lpds and dsrl MUST be separate envs (both define a `stable_baselines3` package).

## Verified
dsrlstudy: Python 3.11.18, pytorch 2.14.1 (conda-forge cuda129 build), torch.cuda.is_available() True on GTX 1650, CUDA matmul works. Not run: pytest/scipy imports beyond install.
Not verified: everything in the upstream specs. conda-lock is not installed, so the explicit file serves as the lock.
Note: env is ~11 GB on disk (CUDA libs).
