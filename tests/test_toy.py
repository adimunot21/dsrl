"""Tests for dsrlstudy.toy. A small shared decoder keeps the suite fast."""
from __future__ import annotations

import pytest
import torch

from dsrlstudy.noise_policy import CountedNoisePolicy, check_determinism
from dsrlstudy.toy import (FlowDecoder, ToyConfig, ToyTask, mode_recovery_stats, toy_grid,
                           train_decoder)


@pytest.fixture(scope="module")
def inside_decoder() -> FlowDecoder:
    return train_decoder(ToyConfig("inside", 0.15, 0.2, train_steps=1500), seed=0)


def test_grid_has_27_unique_configs() -> None:
    grid = toy_grid()
    assert len(grid) == 27
    assert len({c.hash() for c in grid}) == 27
    assert {c.support for c in grid} == {"inside", "edge", "outside"}


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ToyConfig(support="far")
    with pytest.raises(ValueError):
        ToyConfig(weakness=1.5)


def test_goal_distance_to_nearest_mode_matches_support() -> None:
    gen = torch.Generator().manual_seed(0)
    for support, expected in [("inside", 0.0), ("edge", 0.5), ("outside", 1.5)]:
        task = ToyTask(ToyConfig(support))
        states = task.sample_states(200, gen)
        nearest = (task.mode_centres(states) - task.goal(states).unsqueeze(1)).norm(dim=-1).min(dim=1).values
        assert torch.allclose(nearest, torch.full_like(nearest, expected), atol=1e-5)


def test_rewards_dense_and_sparse() -> None:
    task = ToyTask(ToyConfig("inside"))
    states = task.sample_states(10, torch.Generator().manual_seed(1))
    goal = task.goal(states)
    assert torch.allclose(task.reward(states, goal), torch.zeros(10))
    assert (task.reward(states, goal, sparse=True) == 1).all()
    assert (task.reward(states, goal + 0.5, sparse=True) == 0).all()
    assert torch.allclose(task.reward(states, goal + torch.tensor([0.3, 0.4])), torch.full((10,), -0.5))
    with pytest.raises(ValueError):
        task.reward(states, goal[:5])


def test_mixture_mass_matches_weakness() -> None:
    task = ToyTask(ToyConfig("inside", 0.05, 0.2))
    gen = torch.Generator().manual_seed(0)
    states = task.sample_states(1, gen).expand(20000, -1)
    actions = task.sample_actions(states, gen)
    assign = torch.cdist(actions, task.mode_centres(states[:1])[0]).argmin(1)
    assert abs((assign == 0).float().mean().item() - 0.2) < 0.02


def test_decoder_is_frozen_deterministic_and_counted(inside_decoder: FlowDecoder) -> None:
    assert not inside_decoder.training
    assert not any(p.requires_grad for p in inside_decoder.parameters())
    policy = CountedNoisePolicy(inside_decoder)
    gen = torch.Generator().manual_seed(0)
    task = ToyTask(inside_decoder.config)
    states = task.sample_states(32, gen)
    check_determinism(policy, states, policy.sample_prior_noise(32, gen))
    assert policy.decoder_calls == 0
    policy.decode(states, policy.sample_prior_noise(32, gen))
    assert policy.decoder_calls == 32


def test_decoder_matches_mixture(inside_decoder: FlowDecoder) -> None:
    stats = mode_recovery_stats(inside_decoder, ToyTask(inside_decoder.config), n_states=5, n_noise=4000)
    assert stats["max_mass_error"] < 0.08
    assert 0.6 < stats["std_ratio"] < 1.6


def test_training_repeatable_for_seed() -> None:
    cfg = ToyConfig("inside", 0.15, 0.2, train_steps=20, batch_size=64)
    a, b = train_decoder(cfg, 3), train_decoder(cfg, 3)
    assert a.config_hash == b.config_hash
    obs, noise = torch.zeros(4, 2), torch.ones(4, 2)
    assert torch.allclose(a.decode(obs, noise), b.decode(obs, noise), atol=1e-3)
