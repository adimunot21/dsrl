"""Stage-1 analysis (prereg 002 / configs/gym_budget.json): arm A (released) vs arm C (truncation fix), Hopper, seeds 0-9.
Metric per run: mean eval reward over the last 10% of eval checkpoints. Bootstrap CIs over seeds (10,000 resamples, fixed seed).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CONFIG = {"root": "results/gym_stage1/runs/stage1", "n_boot": 10_000, "boot_seed": 20261030,
          "tail_fraction": 0.10, "out": "results/gym_stage1_analysis.json"}


def final_return(path: Path) -> tuple[float, float, int]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["deterministic"] is not None]
    rows.sort(key=lambda r: r["timesteps"])
    n_tail = max(1, int(np.ceil(len(rows) * CONFIG["tail_fraction"])))
    return float(np.mean([r["reward"] for r in rows[-n_tail:]])), rows[0]["reward"], len(rows)


def boot_ci(x: np.ndarray, gen: np.random.Generator) -> list[float]:
    m = [x[gen.integers(0, len(x), len(x))].mean() for _ in range(CONFIG["n_boot"])]
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


def main() -> None:
    gen = np.random.default_rng(CONFIG["boot_seed"])
    root = Path(CONFIG["root"])
    final, init = {}, {}
    for arm in "AC":
        res = [final_return(root / f"{arm}_s{s}" / "eval.jsonl") for s in range(10)]
        final[arm] = np.array([r[0] for r in res]); init[arm] = np.array([r[1] for r in res])
        print(arm, "n_evals", {r[2] for r in res})
    out = {}
    for arm in "AC":
        out[arm] = {"final_per_seed": final[arm].tolist(), "mean": float(final[arm].mean()), "ci95": boot_ci(final[arm], gen),
                    "initial_eval_mean": float(init[arm].mean())}
        gain = final[arm] - init[arm]
        out[arm]["gain_over_initial_mean"] = float(gain.mean()); out[arm]["gain_ci95"] = boot_ci(gain, gen)
    diff = final["C"] - final["A"]
    out["C_minus_A_paired"] = {"mean": float(diff.mean()), "ci95": boot_ci(diff, gen), "n_seeds_C_greater": int((diff > 0).sum())}
    Path(CONFIG["out"]).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items() if a != "final_per_seed"})
    print("A per seed", np.round(final["A"]).tolist()); print("C per seed", np.round(final["C"]).tolist())


if __name__ == "__main__":
    main()
