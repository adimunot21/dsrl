"""Forward-pass noise-search probe (prereg 001): J0, search ceiling C_N, headroom, kNN entropy, variance.

Every probe takes a `CountedNoisePolicy` and returns a `ProbeReading` that records how many decoder
calls (rows) it used, so probe cost is always reported. Reward evaluation is free in the toy task
(closed form) -- for real environments it is not (see prereg threats to validity).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from dsrlstudy.noise_policy import CountedNoisePolicy
from dsrlstudy.toy import ToyTask

logger = logging.getLogger(__name__)

CONFIG = {
    "n_states": 2000,                   # prereg: 2,000 held-out start states
    "ceiling_n_set": (16, 64, 256, 1024),  # prereg N set
    "primary_n": 256,                   # prereg: H_probe = C_256, fixed before results
    "ref_n": 100_000,                   # prereg C_ref
    "knn_k": 5,                         # prereg k
    "entropy_samples_per_state": 256,   # enough points for a stable k=5 estimate in 2-D; cost = n_states*256 calls
    "chunk_rows": 262_144,              # decoder batch cap to bound memory (rows)
}


@dataclass(frozen=True)
class ProbeReading:
    value: float
    decoder_calls: int
    n_states: int
    n_noise: int
    per_state: np.ndarray | None = None


def _decode_chunked(policy: CountedNoisePolicy, states: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """states [R,2], noise [R,2] -> actions [R,2], in bounded chunks."""
    chunk = CONFIG["chunk_rows"]
    return torch.cat([policy.decode(states[i:i + chunk], noise[i:i + chunk])
                      for i in range(0, states.shape[0], chunk)])


def base_return(policy: CountedNoisePolicy, task: ToyTask, n_states: int, generator: torch.Generator,
                n_noise_per_state: int = 1, sparse: bool = False) -> ProbeReading:
    """J0: mean return with w ~ N(0,I). Prereg reading: 1 noise per state over n_states held-out states."""
    calls_before = policy.decoder_calls
    states = task.sample_states(n_states, generator).repeat_interleave(n_noise_per_state, dim=0)
    noise = policy.sample_prior_noise(states.shape[0], generator)
    rewards = task.reward(states, _decode_chunked(policy, states, noise), sparse=sparse)
    return ProbeReading(rewards.mean().item(), policy.decoder_calls - calls_before, n_states, 1,
                        rewards.view(n_states, n_noise_per_state).mean(1).numpy())


def search_ceiling_multi(policy: CountedNoisePolicy, task: ToyTask, n_noise_set: Sequence[int], n_states: int,
                         generator: torch.Generator, sparse: bool = False,
                         box_bound: float | None = None) -> Dict[int, ProbeReading]:
    """C_N for several N at once using nested noise sets (first N of max(N) draws per state).

    box_bound: if given, prior noise is clipped to [-box_bound, box_bound] (amendment 001c, C_N^box).

    Cost is n_states * max(N) decoder calls, charged once (reported on each reading as the shared total).
    """
    n_max = max(n_noise_set)
    calls_before = policy.decoder_calls
    states = task.sample_states(n_states, generator)
    best_by_n: Dict[int, torch.Tensor] = {n: torch.full((n_states,), -math.inf) for n in n_noise_set}
    # process states in blocks so rows-per-call stays bounded
    block = max(1, CONFIG["chunk_rows"] // n_max)
    for start in range(0, n_states, block):
        s_block = states[start:start + block]
        rep = s_block.repeat_interleave(n_max, dim=0)
        noise = policy.sample_prior_noise(rep.shape[0], generator)
        if box_bound is not None:
            noise = noise.clamp(-box_bound, box_bound)
        reward = task.reward(rep, policy.decode(rep, noise), sparse=sparse).view(s_block.shape[0], n_max)
        for n in n_noise_set:
            best_by_n[n][start:start + block] = reward[:, :n].max(dim=1).values
    total_calls = policy.decoder_calls - calls_before
    return {n: ProbeReading(best_by_n[n].mean().item(), total_calls, n_states, n, best_by_n[n].numpy())
            for n in n_noise_set}


def search_ceiling(policy: CountedNoisePolicy, task: ToyTask, n_noise: int, n_states: int,
                   generator: torch.Generator, sparse: bool = False) -> ProbeReading:
    """C_N: per state draw N noises, decode, take max reward; average over states."""
    if n_noise <= 0 or n_states <= 0:
        raise ValueError("n_noise and n_states must be positive")
    return search_ceiling_multi(policy, task, [n_noise], n_states, generator, sparse)[n_noise]


def headroom(ceiling: float, base: float) -> float:
    """H_probe = C_N - J0 (primary uses N = CONFIG['primary_n'])."""
    return ceiling - base


def knn_entropy(samples: np.ndarray, k: int = CONFIG["knn_k"]) -> float:
    """Kozachenko-Leonenko differential entropy (nats): psi(n) - psi(k) + log V_d + (d/n) sum log r_k(i)."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be [n,d], got {samples.shape}")
    n, d = samples.shape
    if n <= k:
        raise ValueError(f"need n > k, got n={n}, k={k}")
    distances, _ = cKDTree(samples).query(samples, k=k + 1)  # column 0 is the point itself
    radius = np.maximum(distances[:, k], 1e-12)  # floor guards duplicate points (log 0)
    log_unit_ball = (d / 2) * math.log(math.pi) - gammaln(d / 2 + 1)
    return float(digamma(n) - digamma(k) + log_unit_ball + d * np.log(radius).mean())


def _per_state_samples(policy: CountedNoisePolicy, task: ToyTask, n_states: int, n_samples: int,
                       generator: torch.Generator) -> torch.Tensor:
    states = task.sample_states(n_states, generator).repeat_interleave(n_samples, dim=0)
    noise = policy.sample_prior_noise(states.shape[0], generator)
    return _decode_chunked(policy, states, noise).view(n_states, n_samples, -1)


def base_action_entropy(policy: CountedNoisePolicy, task: ToyTask, n_states: int, generator: torch.Generator,
                        n_samples: int = CONFIG["entropy_samples_per_state"]) -> ProbeReading:
    """H_ent: per-state kNN entropy of base actions (w~N(0,I)), averaged over states."""
    before = policy.decoder_calls
    actions = _per_state_samples(policy, task, n_states, n_samples, generator).numpy()
    per_state = np.array([knn_entropy(actions[i]) for i in range(n_states)])
    return ProbeReading(float(per_state.mean()), policy.decoder_calls - before, n_states, n_samples, per_state)


def action_variance(policy: CountedNoisePolicy, task: ToyTask, n_states: int, generator: torch.Generator,
                    n_samples: int = CONFIG["entropy_samples_per_state"]) -> ProbeReading:
    """Secondary predictor: per-state total variance (trace of covariance) of base actions, averaged."""
    before = policy.decoder_calls
    actions = _per_state_samples(policy, task, n_states, n_samples, generator)
    per_state = actions.var(dim=1, unbiased=True).sum(dim=-1).numpy()
    return ProbeReading(float(per_state.mean()), policy.decoder_calls - before, n_states, n_samples, per_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from dsrlstudy.toy import ToyConfig, train_decoder

    rng = np.random.default_rng(0)
    for d, sigma in [(2, 1.0), (2, 0.3), (3, 1.0)]:
        est = knn_entropy(rng.normal(0, sigma, size=(20000, d)))
        true = 0.5 * d * math.log(2 * math.pi * math.e * sigma ** 2)
        print(f"Gaussian d={d} sigma={sigma}: kNN={est:.4f} true={true:.4f} err={est - true:+.4f}")

    cfg = ToyConfig("inside", 0.15, 0.2)
    task, policy = ToyTask(cfg), CountedNoisePolicy(train_decoder(cfg, seed=0))
    gen = torch.Generator().manual_seed(123)
    j0 = base_return(policy, task, CONFIG["n_states"], gen)
    print(f"J0={j0.value:.4f} calls={j0.decoder_calls}")
    ceilings = search_ceiling_multi(policy, task, CONFIG["ceiling_n_set"], CONFIG["n_states"], gen)
    for n, r in ceilings.items():
        print(f"C_{n}={r.value:.4f}")
    print(f"ceiling set shared cost = {next(iter(ceilings.values())).decoder_calls} decoder calls")
    print(f"H_probe(N=256) = {headroom(ceilings[256].value, j0.value):.4f}")
    ent = base_action_entropy(policy, task, CONFIG["n_states"], gen)
    var = action_variance(policy, task, CONFIG["n_states"], gen)
    print(f"H_ent={ent.value:.4f} (calls {ent.decoder_calls})  var={var.value:.4f} (calls {var.decoder_calls})")
    ref = search_ceiling(policy, task, CONFIG["ref_n"], 100, gen)
    print(f"C_ref (N=100000, 100 states)={ref.value:.4f} calls={ref.decoder_calls}")
