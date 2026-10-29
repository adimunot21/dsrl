"""Pre-registered analysis of the cloud toy grid (prereg 001 + amendments 001a, 001c, 001d).

Evaluates exactly the decisions written before the grid ran:
  H3 primary : Spearman rho(H_probe_256, G) across the 27 entropy-0.0 configs, bootstrap CI over configs;
               supported only if rho >= 0.7 AND CI lower bound > rho(H_ent, G) point estimate. Kill rule: rho < 0.5.
  H1 (i)     : fraction of (config, seed) runs with G > C_ref - J0 + 0.05 (should be < 5%).
  H1 (ii)    : on "unreachable" configs (mean reachability gap > 0.3, amendment 001a) vs "inside" configs.
Secondary / exploratory (labelled as such): variance and box-ceiling predictors, the minus-dim subset.
Writes results/toy_grid_analysis.json and prints a table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CONFIG = {
    "runs_dir": "results/cloud_grid/runs/toy",   # synced from the pod
    "out_path": "results/toy_grid_analysis.json",
    "n_boot": 10_000,                            # prereg: 10,000 resamples
    "boot_seed": 20261029,                       # fixed so the CI is reproducible
    "rho_support": 0.7,                          # prereg H3 threshold
    "rho_kill": 0.5,                             # prereg kill rule
    "h1_exceed_margin": 0.05,                    # prereg H1(i)
    "h1_exceed_max_fraction": 0.05,
    "gap_unreachable": 0.3,                      # amendment 001a
    "min_unreachable_configs": 6,                # amendment 001a
}
NAME_PATTERN = re.compile(r"(inside|edge|outside)_c([0-9.]+)_w([0-9.]+)")


def load_runs(runs_dir: Path) -> dict[str, list[dict]]:
    """Group results by target-entropy tag; skip the no-box benchmark replicate."""
    groups: dict[str, list[dict]] = {"0.0": [], "-2.0": []}
    for path in sorted(runs_dir.glob("toy_main_*/results.json")):
        if path.parent.name.endswith("_nobox"):
            continue
        entropy_tag = re.search(r"_ent(-?[0-9.]+)_", path.parent.name).group(1)
        record = json.loads(path.read_text())
        match = NAME_PATTERN.fullmatch(record["toy_name"])
        record["support"], record["concentration"], record["weakness"] = match.group(1), float(match.group(2)), float(match.group(3))
        record["run_name"] = path.parent.name
        groups.setdefault(entropy_tag, []).append(record)
    return groups


def config_table(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        probes = rec["probes"]
        gains = np.array(rec["per_seed_gain"])
        rows.append({
            "config": rec["toy_name"], "support": rec["support"],
            "gain_mean": float(gains.mean()), "gain_per_seed": gains.tolist(),
            "H_probe_256": probes["H_probe_256"], "H_probe_box_256": probes["H_probe_box_256"],
            "H_ref": probes["C_ref"]["value"] - probes["J0"], "J0": probes["J0"],
            "H_ent": probes["H_ent"], "action_variance": probes["action_variance"],
            "reach_gap": probes["reachability_gap"]["mean"],
        })
    return rows


def spearman_with_ci(predictor: np.ndarray, gain: np.ndarray, generator: np.random.Generator, n_boot: int) -> dict:
    rho = float(spearmanr(predictor, gain).statistic)
    boot = []
    n = len(gain)
    for _ in range(n_boot):
        idx = generator.integers(0, n, n)
        if np.ptp(predictor[idx]) == 0 or np.ptp(gain[idx]) == 0:
            continue
        boot.append(spearmanr(predictor[idx], gain[idx]).statistic)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"rho": rho, "ci95": [float(low), float(high)], "n_valid_resamples": len(boot)}


def bootstrap_mean_ci(values: np.ndarray, generator: np.random.Generator, n_boot: int) -> list[float]:
    means = [values[generator.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def analyse(rows: list[dict], label: str) -> dict:
    generator = np.random.default_rng(CONFIG["boot_seed"])
    gain = np.array([r["gain_mean"] for r in rows])
    out: dict = {"label": label, "n_configs": len(rows)}
    predictors = {"H_probe_256": "primary", "H_ent": "primary-baseline", "action_variance": "secondary-baseline",
                  "H_probe_box_256": "secondary", "H_ref": "exploratory"}
    out["spearman"] = {}
    for name, role in predictors.items():
        result = spearman_with_ci(np.array([r[name] for r in rows]), gain, generator, CONFIG["n_boot"])
        result["role"] = role
        out["spearman"][name] = result
    primary, entropy = out["spearman"]["H_probe_256"], out["spearman"]["H_ent"]
    out["H3"] = {
        "rho_probe": primary["rho"], "rho_probe_ci95": primary["ci95"], "rho_entropy": entropy["rho"],
        "supported": bool(primary["rho"] >= CONFIG["rho_support"] and primary["ci95"][0] > entropy["rho"]),
        "kill_rule_triggered": bool(primary["rho"] < CONFIG["rho_kill"]),
    }
    # H1(i): ceiling exceeded, per (config, seed)
    exceed = [g > r["H_ref"] + CONFIG["h1_exceed_margin"] for r in rows for g in r["gain_per_seed"]]
    out["H1_i"] = {"fraction_exceeding_C_ref": float(np.mean(exceed)), "n_runs": len(exceed),
                   "passes": bool(np.mean(exceed) < CONFIG["h1_exceed_max_fraction"])}
    # H1(ii): unreachable vs inside
    unreachable = [r for r in rows if r["reach_gap"] > CONFIG["gap_unreachable"]]
    inside = [r for r in rows if r["support"] == "inside"]
    h1ii: dict = {"n_unreachable": len(unreachable), "unreachable": [r["config"] for r in unreachable]}
    if len(unreachable) < CONFIG["min_unreachable_configs"]:
        h1ii["verdict"] = "untestable at this grid (amendment 001a: fewer than 6 unreachable configs)"
    else:
        un_gain = np.array([r["gain_mean"] for r in unreachable])
        in_mean = float(np.mean([r["gain_mean"] for r in inside]))
        ci = bootstrap_mean_ci(un_gain, generator, CONFIG["n_boot"])
        h1ii.update({"mean_gain_unreachable": float(un_gain.mean()), "ci95": ci, "mean_gain_inside": in_mean,
                     "verdict": "supported" if (ci[0] <= 0 <= ci[1] or un_gain.mean() < 0.1 * in_mean) else "not supported"})
    out["H1_ii"] = h1ii
    return out


def main() -> None:
    groups = load_runs(Path(CONFIG["runs_dir"]))
    result: dict = {"counts": {k: len(v) for k, v in groups.items()}}
    rows_main = config_table(groups["0.0"])
    assert len(rows_main) == 27, f"expected 27 entropy-0.0 configs, found {len(rows_main)}"
    result["main_target_entropy_0.0"] = analyse(rows_main, "all 27 configs, target entropy 0.0")
    result["per_config_main"] = [{k: v for k, v in r.items() if k != "gain_per_seed"} for r in rows_main]
    if groups.get("-2.0"):
        rows_sub = config_table(groups["-2.0"])
        result["subset_target_entropy_minus2"] = {
            "note": "9-config subset, exploratory only (n too small for the primary test)",
            "per_config": [{k: v for k, v in r.items() if k != "gain_per_seed"} for r in rows_sub]}
        # paired comparison of gain by entropy target on the same 9 configs
        by_name = {r["config"]: r["gain_mean"] for r in rows_main}
        result["entropy_effect_on_subset"] = [
            {"config": r["config"], "gain_ent0": by_name[r["config"]], "gain_ent_minus2": r["gain_mean"]} for r in rows_sub]
    Path(CONFIG["out_path"]).write_text(json.dumps(result, indent=1))
    h3 = result["main_target_entropy_0.0"]["H3"]
    print(json.dumps({"counts": result["counts"], "H3": h3, "H1_i": result["main_target_entropy_0.0"]["H1_i"],
                      "H1_ii": result["main_target_entropy_0.0"]["H1_ii"]}, indent=1))
    print("\nspearman vs gain (entropy 0.0):")
    for name, s in result["main_target_entropy_0.0"]["spearman"].items():
        print(f"  {name:18s} rho={s['rho']:+.3f}  95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  ({s['role']})")


if __name__ == "__main__":
    main()
