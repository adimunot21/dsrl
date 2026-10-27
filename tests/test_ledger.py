import pytest

from dsrlstudy import ledger
from dsrlstudy.ledger import (RunLedger, budget_status, config_hash,
                              load_ledger, total_cost)


def test_failure_path_records_status(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLedger(tmp_path, "r1", {"a": 1}, seed=1, label="x"):
            raise RuntimeError("boom")
    entry = load_ledger(tmp_path)[0]
    assert entry["status"] == "failed"
    assert entry["error_type"] == "RuntimeError"
    assert (tmp_path / "r1" / "config.json").exists()


def test_success_costs_and_counts(tmp_path):
    with RunLedger(tmp_path, "r2", {"a": 1}, seed=1, label="x",
                   cost_per_hour_usd=3600.0, num_gpus=2, cpu_cores_used=4) as run:
        run.add_counts(env_steps=5, gradient_steps=2, decoder_calls=1)
        run.add_counts(env_steps=5)
        run.log_metric("loss", 1.5, 3)
    entry = load_ledger(tmp_path)[0]
    hours = entry["wall_clock_seconds"] / 3600
    assert entry["status"] == "ok"
    assert entry["counts"] == {"env_steps": 10, "gradient_steps": 2, "decoder_calls": 1}
    assert entry["gpu_hours"] == pytest.approx(2 * hours)
    assert entry["cpu_core_hours"] == pytest.approx(4 * hours)
    assert entry["cost_usd"] == pytest.approx(3600.0 * hours)
    assert entry["metrics"] == [{"name": "loss", "value": 1.5, "step": 3}]


def test_duplicate_run_id_refused(tmp_path):
    with RunLedger(tmp_path, "dup", {}, seed=0, label="x"):
        pass
    with pytest.raises(FileExistsError):
        with RunLedger(tmp_path, "dup", {}, seed=0, label="x"):
            pass
    assert len(load_ledger(tmp_path)) == 1


def test_hash_stable_across_key_order_and_rejects_unserializable():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == \
        config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises(TypeError):
        config_hash({"a": object()})


def test_budget_status_thresholds():
    entries = [{"cost_usd": 79.0}]
    below = budget_status(entries, 100.0)
    assert not below["stop_required"] and below["remaining"] == pytest.approx(21.0)
    assert budget_status([{"cost_usd": 80.0}], 100.0)["stop_required"]
    assert budget_status([{"cost_usd": 95.0}], 100.0)["fraction"] == pytest.approx(0.95)
    with pytest.raises(ValueError):
        budget_status(entries, 0)


def test_total_cost_since():
    entries = [{"cost_usd": 1.0, "start_time": "2026-01-01T00:00:00+00:00"},
               {"cost_usd": 2.0, "start_time": "2026-06-01T00:00:00+00:00"}]
    assert total_cost(entries) == 3.0
    assert total_cost(entries, since="2026-03-01T00:00:00+00:00") == 2.0


def test_no_gpu_fallback(monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    hardware = ledger.capture_hardware()
    assert hardware["gpu"] is None
    assert hardware["logical_cores"] >= 1
    assert "hostname" not in hardware and len(hardware["hostname_hash"]) == 12


def test_git_state_non_repo(tmp_path):
    state = ledger.git_state(tmp_path)
    assert state["head"] is None
    assert ledger.git_state(tmp_path / "missing")["head"] is None
