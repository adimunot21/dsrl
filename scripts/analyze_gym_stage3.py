"""Stage-3 analysis (prereg 002c): arm B (LP-DS as released) vs arm A (DSRL-SAC as released, stage 2), Hopper.
Pairs by seed over the seeds where BOTH arms completed (seeds 10-17; seeds 18-19 of arm B were cut off by the pod hard limit).
Metric: mean of the last 3 evals. Bootstrap over seeds, 10,000 resamples, fixed seed. Decision rule fixed in 002c.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CONFIG = {"a_root": "results/gym_stage2/runs/stage2", "b_root": "results/gym_stage3/runs/stage3", "seeds": list(range(10, 20)),
          "min_final_timestep": 44000, "last_k": 3, "n_boot": 10_000, "boot_seed": 20261030, "out": "results/gym_stage3_analysis.json"}


def load(path: Path) -> list[dict]:
    return sorted((json.loads(l) for l in path.read_text().splitlines() if l.strip()), key=lambda r: r["timesteps"])


def ci(x: np.ndarray, gen: np.random.Generator) -> list[float]:
    m = [x[gen.integers(0, len(x), len(x))].mean() for _ in range(CONFIG["n_boot"])]
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


def main() -> None:
    gen = np.random.default_rng(CONFIG["boot_seed"])
    a_runs = {s: load(Path(CONFIG["a_root"]) / f"A_s{s}" / "eval.jsonl") for s in CONFIG["seeds"]}
    b_runs = {s: load(Path(CONFIG["b_root"]) / f"B_s{s}" / "eval.jsonl") for s in CONFIG["seeds"]}
    complete = [s for s in CONFIG["seeds"] if b_runs[s][-1]["timesteps"] >= CONFIG["min_final_timestep"] and a_runs[s][-1]["timesteps"] >= CONFIG["min_final_timestep"]]
    partial = [s for s in CONFIG["seeds"] if s not in complete]
    metric = lambda rows: float(np.mean([r["reward"] for r in rows[-CONFIG["last_k"]:]]))
    a = np.array([metric(a_runs[s]) for s in complete]); b = np.array([metric(b_runs[s]) for s in complete])
    diff = b - a
    out = {"seeds_used": complete, "seeds_partial_excluded": partial,
           "partial_last_timestep": {str(s): b_runs[s][-1]["timesteps"] for s in partial},
           "A": {"mean": float(a.mean()), "ci95": ci(a, gen), "per_seed": a.tolist()},
           "B": {"mean": float(b.mean()), "ci95": ci(b, gen), "per_seed": b.tolist()},
           "B_minus_A": {"mean": float(diff.mean()), "ci95": ci(diff, gen), "n_seeds_B_greater": int((diff > 0).sum())},
           "initial_eval_mean": {"A": float(np.mean([a_runs[s][0]["reward"] for s in complete])), "B": float(np.mean([b_runs[s][0]["reward"] for s in complete]))},
           "B_gain_over_initial": float((b - np.array([b_runs[s][0]["reward"] for s in complete])).mean())}
    lo, hi = out["B_minus_A"]["ci95"]
    out["verdict_prereg_002c"] = "B (LP-DS) better" if lo > 0 else "A (DSRL) better" if hi < 0 else "no detectable difference"
    Path(CONFIG["out"]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("A", "B")}, indent=1)); print("A", round(out["A"]["mean"], 1), out["A"]["ci95"]); print("B", round(out["B"]["mean"], 1), out["B"]["ci95"])


if __name__ == "__main__":
    main()
