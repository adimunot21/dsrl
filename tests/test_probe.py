"""Tests for dsrlstudy.probe, using an analytic (untrained) decoder for speed and exact answers."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsrlstudy import probe
from dsrlstudy.noise_policy import CountedNoisePolicy
from dsrlstudy.toy import ToyConfig, ToyTask


class IdentityDecoder:
    """action = goal-anchored noise: a = best-mode-centre + 0.5 * w. Deterministic, no training."""

    noise_shape = (2,)
    deterministic = True

    def __init__(self, task: ToyTask) -> None:
        self.task = task

    def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.task.goal(obs) + 0.5 * noise


@pytest.fixture()
def setup():
    task = ToyTask(ToyConfig("inside"))
    return task, CountedNoisePolicy(IdentityDecoder(task)), torch.Generator().manual_seed(0)


def test_knn_entropy_gaussian() -> None:
    rng = np.random.default_rng(0)
    for dim, sigma in [(2, 1.0), (2, 0.3), (3, 2.0)]:
        true = 0.5 * dim * math.log(2 * math.pi * math.e * sigma ** 2)
        assert abs(probe.knn_entropy(rng.normal(0, sigma, (20000, dim))) - true) < 0.05


def test_knn_entropy_validation() -> None:
    with pytest.raises(ValueError):
        probe.knn_entropy(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        probe.knn_entropy(np.zeros(10))
    assert np.isfinite(probe.knn_entropy(np.zeros((50, 2))))  # duplicates do not give -inf/nan


def test_base_return_and_cost(setup) -> None:
    task, policy, gen = setup
    reading = probe.base_return(policy, task, 4000, gen)
    # E||0.5 w|| for 2-D standard normal = 0.5 * sqrt(pi/2)
    assert reading.value == pytest.approx(-0.5 * math.sqrt(math.pi / 2), abs=0.03)
    assert reading.decoder_calls == 4000 == policy.decoder_calls


def test_ceiling_monotone_and_cost(setup) -> None:
    task, policy, gen = setup
    readings = probe.search_ceiling_multi(policy, task, (16, 64, 256, 1024), 300, gen)
    values = [readings[n].value for n in (16, 64, 256, 1024)]
    assert values == sorted(values) and values[-1] <= 0.0
    assert readings[16].decoder_calls == 300 * 1024 == policy.decoder_calls


def test_single_ceiling_cost_and_headroom(setup) -> None:
    task, policy, gen = setup
    j0 = probe.base_return(policy, task, 500, gen)
    c256 = probe.search_ceiling(policy, task, 256, 500, gen)
    assert c256.decoder_calls == 500 * 256
    assert probe.headroom(c256.value, j0.value) > 0.2
    with pytest.raises(ValueError):
        probe.search_ceiling(policy, task, 0, 10, gen)


def test_entropy_and_variance_of_known_decoder(setup) -> None:
    task, policy, gen = setup
    ent = probe.base_action_entropy(policy, task, 50, gen, n_samples=512)
    var = probe.action_variance(policy, task, 50, gen, n_samples=512)
    assert ent.value == pytest.approx(math.log(2 * math.pi * math.e * 0.25), abs=0.1)
    assert var.value == pytest.approx(0.5, abs=0.05)  # trace cov = 2 * 0.25
    assert ent.decoder_calls == var.decoder_calls == 50 * 512
