"""Tests for the ensemble steerer (identity decoder: action = w). Heavy tests use CUDA+graph when present."""
import time

import pytest
import torch

from dsrlstudy.noise_policy import CountedNoisePolicy
from dsrlstudy.steerer import train_steerer
from dsrlstudy.steerer_batched import _Ensemble, benchmark_ms_per_step, train_steerer_ensemble
from dsrlstudy.steerer_batched import CONFIG as BCONFIG

TARGET = torch.tensor([1.5, -1.0])
HAS_CUDA = torch.cuda.is_available()
FAST = {"total_env_steps": 1600, "utd": 2, "learning_starts": 300, "log_every": 800, "envs_per_step": 8,
        "hidden": 64, "n_layers": 2, "batch": 64, "lr": 1e-3, "b_w": 3.0}
GPU = {"device": "cuda", "cuda_graph": True} if HAS_CUDA else {}
TINY = {"total_env_steps": 160, "utd": 1, "learning_starts": 40, "log_every": 80, "envs_per_step": 8,
        "hidden": 16, "n_layers": 1, "batch": 16, "replay_size": 512}


class _Identity:
    noise_shape = (2,)

    def decode(self, obs, noise):
        return noise


def _states(n, gen):
    return torch.randn(n, 1, generator=gen)


def _reward(s, a):
    return -(a - TARGET).norm(dim=-1)


def _eval(res):
    w = res.act(torch.zeros(64, 1), deterministic=True)
    return _reward(None, w).mean().item()


def _ens(seeds, cfg, reward=_reward, **extra):
    policy = CountedNoisePolicy(_Identity())
    return policy, train_steerer_ensemble(policy, _states, reward, {**cfg, **extra}, seeds)


def test_independence_perturbed_reward_member0():
    """Member 1's parameters are bitwise unchanged when only member 0's reward changes."""
    S = 2

    def make(perturb):
        calls = [0]

        def reward(s, a):
            member = calls[0] % S  # reward_fn is called once per member per iteration, in order
            calls[0] += 1
            r = _reward(s, a)
            return r + 5.0 * torch.sin(a[:, 0] * 3.0) if (perturb and member == 0) else r
        return reward

    _, base = _ens([0, 1], TINY, make(False), device="cpu")
    _, pert = _ens([0, 1], TINY, make(True), device="cpu")
    for m in (0, 1):
        pass
    p0 = [p for p in base[0].actor.parameters()]
    q0 = [p for p in pert[0].actor.parameters()]
    p1 = [p for p in base[1].actor.parameters()]
    q1 = [p for p in pert[1].actor.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(p1, q1)), "member 1 changed when only member 0's reward changed"
    assert any(not torch.equal(a, b) for a, b in zip(p0, q0)), "member 0 should have changed"
    print("\n[independence] member 1 bitwise identical; member 0 differs")


def test_ensemble_member_matches_solo_numerically():
    """Sum-of-losses => one Adam over stacked tensors == independent Adams. Member 0 of [0,1] must track
    the same member run alone ([0]) up to float noise (bmm batch size changes summation order only)."""
    cfg = {**BCONFIG, **TINY, "device": "cpu"}

    def run(seeds):
        ens = _Ensemble(seeds, 1, 2, cfg, torch.device("cpu"))
        g = torch.Generator().manual_seed(7)
        n = 64
        obs, w = torch.randn(1, n, 1, generator=g), (torch.rand(1, n, 2, generator=g) * 2 - 1) * 3
        r = torch.randn(1, n, generator=g)
        ens.add(obs.expand(len(seeds), -1, -1).contiguous(), w.expand(len(seeds), -1, -1).contiguous(),
                r.expand(len(seeds), -1).contiguous())
        for _ in range(25):
            ens.step()
        return ens.actor.weights[0][0].detach().clone(), ens.critic.weights[0][0].detach().clone()

    solo, duo = run([0]), run([0, 1])
    diffs = [(a - b).abs().max().item() for a, b in zip(solo, duo)]
    print(f"\n[solo vs ensemble member] max |param diff| after 25 steps: actor {diffs[0]:.2e}, critic {diffs[1]:.2e}")
    assert max(diffs) < 1e-4


def test_decoder_calls_equal_env_steps_per_member():
    policy, res = _ens([3, 4, 5], TINY, total_env_steps=1003 // 8 * 8 + 3, device="cpu")  # not a multiple of 8
    n = TINY["total_env_steps"] if False else 1003 // 8 * 8 + 3
    for r in res:
        assert r.env_steps == r.decoder_calls == n
        assert r.gradient_steps > 0
    assert policy.decoder_calls == 3 * n


def test_known_answer_ensemble_S4():
    t0 = time.time()
    _, res = _ens([0, 1, 2, 3], FAST, **GPU)
    vals = [_eval(r) for r in res]
    print(f"\n[known-answer S=4] det. rewards {[round(v, 4) for v in vals]} (optimum 0.0) in {time.time() - t0:.0f}s")
    assert min(vals) > -0.15


def test_equivalence_with_single_steerer_in_box():
    """Statistical (not bitwise) equivalence: S=1 ensemble vs steerer.py, final deterministic reward within 0.05."""
    cfg = {**FAST}
    _, ens = _ens([0], cfg, **GPU)
    single = train_steerer(CountedNoisePolicy(_Identity()), _states, _reward, cfg, 0)
    e, s = _eval(ens[0]), _eval(single)
    print(f"\n[equivalence] ensemble S=1: {e:.4f}   steerer.py: {s:.4f}   |diff| {abs(e - s):.4f}")
    assert abs(e - s) < 0.05


def test_seed_members_differ_and_seed_reproducible():
    _, a = _ens([0, 1], TINY, device="cpu")
    _, b = _ens([0, 1], TINY, device="cpu")
    obs = torch.randn(8, 1)
    assert torch.equal(a[0].act(obs), b[0].act(obs))
    assert not torch.equal(a[0].act(obs), a[1].act(obs))


@pytest.mark.skipif(not HAS_CUDA, reason="needs cuda")
def test_cuda_graph_matches_eager():
    cfg = {**BCONFIG, **TINY}

    def run(graph):
        ens = _Ensemble([0, 1], 1, 2, {**cfg, "device": "cuda", "cuda_graph": graph}, torch.device("cuda"))
        g = torch.Generator().manual_seed(7)
        ens.add(torch.randn(2, 64, 1, generator=g).cuda(), (torch.rand(2, 64, 2, generator=g) * 6 - 3).cuda(),
                torch.randn(2, 64, generator=g).cuda())
        for _ in range(25):
            ens.step()
        return ens.actor.weights[0].detach().clone()

    diff = (run(True) - run(False)).abs().max().item()
    print(f"\n[graph vs eager] max |param diff| {diff:.2e}")
    assert diff < 1e-4


def test_speed_report():
    rows = []
    for device in ["cpu"] + (["cuda"] if HAS_CUDA else []):
        for S in (1, 4, 10):
            for graph in ([False, True] if device == "cuda" else [False]):
                n = 15 if device == "cpu" else 100
                ms = benchmark_ms_per_step(S, device, n_steps=n, warmup=5 if device == "cpu" else 20, cuda_graph=graph)
                rows.append((device, graph, S, ms))
                print(f"\n[speed] {device:4s} graph={graph!s:5s} S={S:2d}: {ms:8.2f} ms/step  {ms / S:7.2f} ms/member-step")
    assert all(r[3] > 0 for r in rows)
