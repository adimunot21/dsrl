"""Unattended session 1 for prereg 001 / amendment 001b (order fixed in 001b item 5).

tune 3 learning rates -> pick the best by mean tuning-seed return -> outside/0.15/0.2 at full budget
-> analytic bound-limited check. A failed step is logged and the sequence continues; nothing is
re-run or re-selected on reported-seed results.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(Path.home() / "miniforge3/envs/dsrlstudy/bin/python")
LEARNING_RATES = [1e-4, 3e-4, 1e-3]  # 001b: the 3 tuning trials
LOG = Path("runs/overnight_001.log")


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def run_step(extra_args: list[str]) -> int:
    command = ["nice", "-n", "10", PYTHON, "scripts/run_toy_config.py", *extra_args]
    log("START " + " ".join(extra_args))
    started = time.time()
    with LOG.open("a") as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "PYTHONPATH": "."}).returncode
    log(f"END   code={code} elapsed={(time.time() - started) / 60:.1f} min")
    return code


def tune_score(lr: float) -> float | None:
    path = Path(f"runs/toy/toy_tune_inside_0.15_0.2_lr{lr}/results.json")
    if not path.exists():
        return None
    return sum(json.loads(path.read_text())["per_seed_return"]) / len(json.loads(path.read_text())["per_seed_return"])


def main() -> None:
    Path("runs").mkdir(exist_ok=True)
    scores: dict[float, float] = {}
    for lr in LEARNING_RATES:
        run_step(["--mode", "tune", "--lr", str(lr)])
        score = tune_score(lr)
        if score is not None:
            scores[lr] = score
        log(f"tune lr={lr} mean tuning return={score}")
    if not scores:
        log("ABORT: all tuning runs failed")
        sys.exit(1)
    best_lr = max(scores, key=scores.get)
    log(f"selected lr={best_lr} from {scores}")
    run_step(["--mode", "main", "--support", "outside", "--concentration", "0.15", "--weakness", "0.2", "--lr", str(best_lr)])
    run_step(["--mode", "bound", "--lr", str(best_lr)])
    log("SESSION DONE")


if __name__ == "__main__":
    main()
