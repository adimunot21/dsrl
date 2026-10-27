"""Known-case tests for the SAC steerer (identity decoder: action = w)."""
import math

import pytest
import torch

from dsrlstudy.noise_policy import CountedNoisePolicy
from dsrlstudy.steerer import train_steerer

TARGET = torch.tensor([1.5, -1.0])
FAST = {"total_env_steps": 2000, "utd": 2, "learning_starts": 300, "log_every": 1000, "envs_per_step": 8,
        "hidden": 64, "n_layers": 2, "batch": 64, "lr": 1e-3}  # small nets: tests must be quick on a shared box


class _Identity:
    noise_shape = (2,)

    def decode(self, obs, noise):
        return noise


def _sample_states(n, gen):
    return torch.randn(n, 1, generator=gen)


def _reward(s, a):
    return -(a - TARGET).norm(dim=-1)


def _run(b_w, seed=0, **extra):
    policy = CountedNoisePolicy(_Identity())
    res = train_steerer(policy, _sample_states, _reward, {**FAST, "b_w": b_w, **extra}, seed)
    return policy, res


def _eval_reward(res):
    w = res.act(torch.zeros(64, 1), deterministic=True)
    assert w.abs().max() <= res.b_w + 1e-6
    return _reward(None, w).mean().item()


def test_reaches_optimum_when_target_in_box():
    _, res = _run(b_w=3.0)
    value = _eval_reward(res)
    print(f"\n[in-box] reward {value:.4f} (optimum 0.0)")
    assert value > -0.15


def test_saturates_at_bound_limited_optimum():
    b_w = 0.8  # box [-0.8,0.8]^2, target (1.5,-1): nearest point (0.8,-0.8)
    analytic = -math.hypot(1.5 - 0.8, -1.0 + 0.8)
    _, res = _run(b_w=b_w, utd=4)  # tanh saturation at the bound is slow to learn
    value = _eval_reward(res)
    print(f"\n[bound-limited] reward {value:.4f} analytic {analytic:.4f}")
    assert abs(value - analytic) < 0.15
    assert value <= analytic + 1e-3  # cannot beat the analytic optimum


def test_decoder_calls_equal_env_steps():
    policy, res = _run(b_w=2.0, total_env_steps=1003, learning_starts=100)  # not a multiple of envs_per_step
    assert res.env_steps == 1003 == res.decoder_calls == policy.decoder_calls
    assert res.gradient_steps > 0


def test_determinism_same_seed():
    _, a = _run(b_w=2.0, total_env_steps=1500, learning_starts=200)
    _, b = _run(b_w=2.0, total_env_steps=1500, learning_starts=200)
    _, c = _run(b_w=2.0, seed=1, total_env_steps=1500, learning_starts=200)
    obs = torch.randn(16, 1)
    assert torch.equal(a.act(obs), b.act(obs))
    assert not torch.equal(a.act(obs), c.act(obs))
    assert [h["mean_reward"] for h in a.history] == [h["mean_reward"] for h in b.history]
