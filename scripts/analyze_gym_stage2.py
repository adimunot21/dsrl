"""Stage-2 analysis (prereg 002b): dense-eval replicate of arm A vs arm C, Hopper, seeds 10-19.
Primary metric: mean eval reward of the last 3 evals. Secondary: mean over all evals after step 0; pooled with stage 1 (seeds 0-9, last eval).
Bootstrap CIs over seeds (10,000 resamples, fixed seed). Decision rule fixed in 002b: CI of C - A entirely below 0 => replicated.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CONFIG = {"root": "results/gym_stage2/runs/stage2", "seeds": list(range(10, 20)), "last_k": 3, "n_boot": 10_000,
          "boot_seed": 20261030, "out": "results/gym_stage2_analysis.json", "stage1": "results/gym_stage1_analysis.json"}


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["timesteps"])


def boot_ci(x: np.ndarray, gen: np.random.Generator) -> list[float]:
    means = [x[gen.integers(0, len(x), len(x))].mean() for _ in range(CONFIG["n_boot"])]
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main() -> None:
    gen = np.random.default_rng(CONFIG["boot_seed"])
    metrics: dict[str, dict[str, np.ndarray]] = {"last3": {}, "auc": {}, "initial": {}}
    for arm in "AC":
        runs = [load(Path(CONFIG["root"]) / f"{arm}_s{s}" / "eval.jsonl") for s in CONFIG["seeds"]]
        assert all(r[-1]["timesteps"] >= 44000 for r in runs), "incomplete run"
        metrics["last3"][arm] = np.array([np.mean([x["reward"] for x in r[-CONFIG["last_k"]:]]) for r in runs])
        metrics["auc"][arm] = np.array([np.mean([x["reward"] for x in r[1:]]) for r in runs])
        metrics["initial"][arm] = np.array([r[0]["reward"] for r in runs])
    out: dict = {"n_evals_per_run": len(runs[0])}
    for name in ("last3", "auc"):
        a, c = metrics[name]["A"], metrics[name]["C"]
        diff = c - a
        out[name] = {"A": {"mean": float(a.mean()), "ci95": boot_ci(a, gen), "per_seed": a.tolist()},
                     "C": {"mean": float(c.mean()), "ci95": boot_ci(c, gen), "per_seed": c.tolist()},
                     "C_minus_A": {"mean": float(diff.mean()), "ci95": boot_ci(diff, gen), "n_seeds_C_greater": int((diff > 0).sum())}}
    out["initial_eval_mean"] = {arm: float(metrics["initial"][arm].mean()) for arm in "AC"}
    ci = out["last3"]["C_minus_A"]["ci95"]
    out["verdict_prereg_002b"] = "C worse than A (replicated)" if ci[1] < 0 else "C better than A" if ci[0] > 0 else "no detectable difference"
    s1 = json.loads(Path(CONFIG["stage1"]).read_text())
    pooled = np.array(s1["C"]["final_per_seed"] + out["last3"]["C"]["per_seed"]) - np.array(s1["A"]["final_per_seed"] + out["last3"]["A"]["per_seed"])
    out["secondary_pooled_20_seeds_C_minus_A"] = {"note": "stage1 metric = single final eval, stage2 = last-3 mean; mixed metric",
                                                   "mean": float(pooled.mean()), "ci95": boot_ci(pooled, gen)}
    Path(CONFIG["out"]).write_text(json.dumps(out, indent=1))
    for name in ("last3", "auc"):
        r = out[name]
        print(name, {k: (round(r[k]["mean"], 1), [round(v, 1) for v in r[k]["ci95"]]) for k in r}, "C>A seeds", r["C_minus_A"]["n_seeds_C_greater"])
    print("initial", out["initial_eval_mean"], "| verdict:", out["verdict_prereg_002b"])
    print("pooled", out["secondary_pooled_20_seeds_C_minus_A"])


if __name__ == "__main__":
    main()
