"""Unattended toy grid for prereg 001 / amendment 001c (cloud pod).

Phase A: all 27 configs at target entropy 0.0 (outside/0.15/0.2 is already done -> skipped if its results exist).
Phase B: the 9-config subset {inside,edge,outside} x {0.05,0.15,0.4} at weakness 0.2 at target entropy -2.0.
Order is fixed here, before any grid result is seen. A failed config is logged and the sequence continues.
The $ guard stops launching new configs once estimated spend reaches the cap; pod spend is NOT measured by us,
so the rate below is the user-reported figure.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dsrlstudy.toy import toy_grid

CONFIG = {
    "pod_rate_usd_per_hour": 0.5,   # user-reported RunPod rate, 2026-10-28
    "spend_cap_usd": 15.0,          # well below the $80 stop line; this grid is estimated at ~$10
    "phase_a_target_entropy": 0.0,  # DSRL's setting (001c)
    "phase_b_target_entropy": -2.0, # -dim, subset only (001c)
    "phase_b_weakness": 0.2,
    "phase_b_concentrations": (0.05, 0.15, 0.4),
}
LOG = Path("runs/grid_cloud.log")


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def planned_jobs() -> list[tuple[str, float, float, float]]:
    jobs = [(c.support, c.concentration, c.weakness, CONFIG["phase_a_target_entropy"]) for c in toy_grid()]
    for c in toy_grid():
        if c.weakness == CONFIG["phase_b_weakness"] and c.concentration in CONFIG["phase_b_concentrations"]:
            jobs.append((c.support, c.concentration, c.weakness, CONFIG["phase_b_target_entropy"]))
    return jobs


def result_exists(support: str, conc: float, weak: float, ent: float, tag: str) -> bool:
    return Path(f"runs/toy/toy_main_{support}_{conc}_{weak}_lr0.0003_ent{ent}{tag}/results.json").exists()


def main() -> None:
    Path("runs").mkdir(exist_ok=True)
    started = time.time()
    tag = "_cloud3090"
    jobs = planned_jobs()
    log(f"grid: {len(jobs)} jobs")
    for support, conc, weak, ent in jobs:
        spent = (time.time() - started) / 3600 * CONFIG["pod_rate_usd_per_hour"]
        if spent >= CONFIG["spend_cap_usd"]:
            log(f"STOP: estimated spend ${spent:.2f} reached cap")
            break
        if result_exists(support, conc, weak, ent, tag):
            log(f"SKIP existing {support}/{conc}/{weak} ent={ent}")
            continue
        args = ["--mode", "main", "--support", support, "--concentration", str(conc), "--weakness", str(weak),
                "--lr", "0.0003", "--target-entropy", str(ent), "--tag", tag]
        log(f"START {' '.join(args)} (grid spend so far ~${spent:.2f})")
        t0 = time.time()
        with LOG.open("a") as handle:
            code = subprocess.run([sys.executable, "scripts/run_toy_config.py", *args], stdout=handle,
                                  stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}).returncode
        log(f"END code={code} {(time.time() - t0) / 60:.1f} min")
    log("GRID DONE -- stop the pod")


if __name__ == "__main__":
    print(len(planned_jobs()), "planned jobs")
    if "--dry-run" not in sys.argv:
        main()
