"""Run ledger: traceable, costed experiment runs.

Every run appends one JSON line to ``<ledger_dir>/ledger.jsonl`` and writes
``<ledger_dir>/<run_id>/config.json``. Stdlib + numpy only.

Entry contract (one JSON object per line):
    run_id str, label str, status 'ok'|'failed', error_type str|None,
    seed int, config_hash str (sha256 hex), start_time/end_time ISO-8601 UTC,
    wall_clock_seconds float, gpu_hours float, cpu_core_hours float,
    cost_usd float, counts {env_steps, gradient_steps, decoder_calls} ints,
    metrics list[{name, value, step}], hardware dict, git dict.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np  # noqa: F401  (used for numpy-scalar coercion in JSON encoding)

logger = logging.getLogger(__name__)

CONFIG: dict[str, Any] = {
    # Subprocess calls must never hang a run; 10 s is ample for nvidia-smi/git.
    "subprocess_timeout_s": 10,
    # Project policy: STOP spending at 80% of the budget cap.
    "default_warn_fraction": 0.8,
    "ledger_filename": "ledger.jsonl",
    "config_filename": "config.json",
    # Hash truncation is NOT applied: full SHA-256 avoids any collision debate.
    "hostname_hash_chars": 12,  # enough to distinguish machines, not reversible
}

COUNTER_NAMES = ("env_steps", "gradient_steps", "decoder_calls")


def _run_cmd(args: list[str], cwd: Optional[str] = None) -> Optional[str]:
    """Run a command, returning stripped stdout or None on any failure."""
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True,
            timeout=CONFIG["subprocess_timeout_s"], check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpu_info() -> Optional[dict[str, Any]]:
    """Query nvidia-smi; None if unavailable or output unparsable."""
    out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        name, mem_mib, driver = [part.strip() for part in out.splitlines()[0].split(",")]
        return {"name": name, "memory_mib": int(float(mem_mib)), "driver": driver}
    except ValueError:
        logger.warning("Unparsable nvidia-smi output: %r", out)
        return None


def capture_hardware() -> dict[str, Any]:
    """Describe the machine. Hostname is hashed, never stored raw."""
    host_hash = hashlib.sha256(socket.gethostname().encode()).hexdigest()
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu_info(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "hostname_hash": host_hash[: CONFIG["hostname_hash_chars"]],
    }


def git_state(repo_path: str | os.PathLike) -> dict[str, Any]:
    """HEAD sha, dirty flag, submodule SHAs. Never raises."""
    state: dict[str, Any] = {"head": None, "dirty": None, "submodules": {}}
    try:
        cwd = str(repo_path)
        head = _run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd)
        if head is None:
            return state
        state["head"] = head
        status = _run_cmd(["git", "status", "--porcelain"], cwd=cwd)
        state["dirty"] = bool(status) if status is not None else None
        subs = _run_cmd(["git", "submodule", "status"], cwd=cwd) or ""
        for line in subs.splitlines():
            parts = line.strip().lstrip("+-U").split()
            if len(parts) >= 2:
                state["submodules"][parts[1]] = parts[0]
    except Exception as exc:  # defensive: contract is "never raise"
        logger.warning("git_state failed: %s", exc)
    return state


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def config_hash(config: dict) -> str:
    """Stable SHA-256 over canonical JSON. Raises TypeError if not serializable."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"),
                           default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunLedger:
    """Context manager recording one costed, traceable run."""

    def __init__(self, ledger_dir: str | os.PathLike, run_id: str, config: dict,
                 seed: int, label: str, cost_per_hour_usd: float = 0.0,
                 num_gpus: int = 0, cpu_cores_used: Optional[int] = None,
                 repo_path: str | os.PathLike = ".") -> None:
        if not run_id or any(sep in run_id for sep in ("/", "\\")) or run_id in (".", ".."):
            raise ValueError(f"invalid run_id: {run_id!r}")
        if cost_per_hour_usd < 0 or num_gpus < 0:
            raise ValueError("cost_per_hour_usd and num_gpus must be >= 0")
        self.ledger_dir = Path(ledger_dir)
        self.run_id = run_id
        self.config = config
        self.seed = int(seed)
        self.label = label
        self.cost_per_hour_usd = float(cost_per_hour_usd)
        self.num_gpus = int(num_gpus)
        self.cpu_cores_used = cpu_cores_used
        self.repo_path = repo_path
        self.counts: dict[str, int] = {name: 0 for name in COUNTER_NAMES}
        self.metrics: list[dict[str, Any]] = []
        self.entry: Optional[dict[str, Any]] = None
        self._config_hash = config_hash(config)  # fail early on bad config
        self._start_perf = 0.0
        self._start_time: Optional[datetime] = None
        self._hardware: dict[str, Any] = {}
        self._git: dict[str, Any] = {}

    def __enter__(self) -> "RunLedger":
        run_dir = self.ledger_dir / self.run_id
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)  # atomic; raises FileExistsError on reuse
        self._hardware = capture_hardware()
        self._git = git_state(self.repo_path)
        (run_dir / CONFIG["config_filename"]).write_text(
            json.dumps(self.config, sort_keys=True, indent=2, default=_json_default),
            encoding="utf-8")
        self._start_time = _utc_now()
        self._start_perf = time.perf_counter()
        logger.info("run %s started (%s)", self.run_id, self.label)
        return self

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.metrics.append({"name": name, "value": float(value), "step": int(step)})

    def add_counts(self, env_steps: int = 0, gradient_steps: int = 0,
                   decoder_calls: int = 0) -> None:
        if min(env_steps, gradient_steps, decoder_calls) < 0:
            raise ValueError("counts must be non-negative")
        self.counts["env_steps"] += int(env_steps)
        self.counts["gradient_steps"] += int(gradient_steps)
        self.counts["decoder_calls"] += int(decoder_calls)

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall_s = time.perf_counter() - self._start_perf
        hours = wall_s / 3600.0
        cores = self.cpu_cores_used if self.cpu_cores_used is not None \
            else (self._hardware.get("logical_cores") or 1)
        self.entry = {
            "run_id": self.run_id,
            "label": self.label,
            "status": "failed" if exc_type is not None else "ok",
            "error_type": exc_type.__name__ if exc_type is not None else None,
            "seed": self.seed,
            "config_hash": self._config_hash,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": _utc_now().isoformat(),
            "wall_clock_seconds": wall_s,
            "gpu_hours": hours * self.num_gpus,
            "cpu_core_hours": hours * cores,
            "cost_usd": hours * self.cost_per_hour_usd,
            "counts": dict(self.counts),
            "metrics": self.metrics,
            "hardware": self._hardware,
            "git": self._git,
        }
        with open(self.ledger_dir / CONFIG["ledger_filename"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.entry, default=_json_default) + "\n")
        logger.info("run %s finished: %s, %.2fs, $%.4f", self.run_id,
                    self.entry["status"], wall_s, self.entry["cost_usd"])
        return False  # never swallow exceptions


def load_ledger(path: str | os.PathLike) -> list[dict]:
    """Load ledger.jsonl (file path, or directory containing it)."""
    p = Path(path)
    if p.is_dir():
        p = p / CONFIG["ledger_filename"]
    if not p.exists():
        return []
    entries = []
    with open(p, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_no}: corrupt ledger line: {exc}") from exc
    return entries


def total_cost(entries: list[dict], since: Optional[datetime | str] = None) -> float:
    """Sum cost_usd, optionally only runs whose start_time >= since."""
    if isinstance(since, str):
        since = datetime.fromisoformat(since)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    total = 0.0
    for entry in entries:
        if since is not None:
            start = entry.get("start_time")
            if not start or datetime.fromisoformat(start) < since:
                continue
        total += float(entry.get("cost_usd", 0.0))
    return total


def budget_status(entries: list[dict], cap_usd: float,
                  warn_fraction: float = CONFIG["default_warn_fraction"]) -> dict[str, Any]:
    """Spend vs cap. stop_required is True once fraction >= warn_fraction."""
    if cap_usd <= 0:
        raise ValueError("cap_usd must be > 0")
    spent = total_cost(entries)
    fraction = spent / cap_usd
    return {"spent": spent, "remaining": cap_usd - spent, "fraction": fraction,
            "stop_required": fraction >= warn_fraction}


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        with RunLedger(tmp, "demo-run", {"lr": 3e-4, "env": "toy"}, seed=0,
                       label="sanity", cost_per_hour_usd=0.5, num_gpus=1,
                       cpu_cores_used=2) as run:
            run.add_counts(env_steps=100, gradient_steps=10, decoder_calls=5)
            run.log_metric("loss", 0.5, step=1)
            time.sleep(0.05)
        entries = load_ledger(tmp)
        print(json.dumps(entries[0], indent=2))
        print(budget_status(entries, cap_usd=1.0))
