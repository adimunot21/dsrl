"""Noise-policy interface: a frozen decoder whose only free input is its initial noise.

Contract (what DSRL needs from a base policy):
  1. The decoder exposes its initial-noise input (shape `noise_shape`).
  2. It is deterministic given (observation, noise): the same pair always gives the same action.
  3. It is frozen: no parameter changes during steering.
Anything that fails (2) cannot be steered and is rejected by `check_determinism`.

The wrapper counts decoder calls so budgets can be matched across methods.
"""
from __future__ import annotations

import logging
from typing import Protocol, Tuple

import torch

logger = logging.getLogger(__name__)

CONFIG = {
    "determinism_trials": 4,  # repeated calls compared exactly; 4 catches RNG use inside the decoder
    "determinism_atol": 0.0,  # exact equality: a deterministic sampler must be bitwise repeatable on one device
}


class NoiseDecoder(Protocol):
    """Anything mapping (obs, noise) -> action deterministically."""

    noise_shape: Tuple[int, ...]

    def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """obs: [batch, obs_dim], noise: [batch, *noise_shape] -> action: [batch, *action_shape]."""
        ...


class CountedNoisePolicy:
    """Wraps a frozen decoder, validates shapes, and counts decoder calls (rows, not batches)."""

    def __init__(self, decoder: NoiseDecoder) -> None:
        self._decoder = decoder
        self.noise_shape: Tuple[int, ...] = tuple(decoder.noise_shape)
        self.decoder_calls: int = 0

    @torch.no_grad()
    def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"obs must be [batch, obs_dim], got {tuple(obs.shape)}")
        if tuple(noise.shape[1:]) != self.noise_shape or noise.shape[0] != obs.shape[0]:
            raise ValueError(
                f"noise must be [{obs.shape[0]}, *{self.noise_shape}], got {tuple(noise.shape)}"
            )
        self.decoder_calls += int(obs.shape[0])
        return self._decoder.decode(obs, noise)

    def sample_prior_noise(self, batch: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(batch, *self.noise_shape, generator=generator)

    def reset_counter(self) -> int:
        calls, self.decoder_calls = self.decoder_calls, 0
        return calls


def check_determinism(policy: CountedNoisePolicy, obs: torch.Tensor, noise: torch.Tensor) -> None:
    """Raise if the decoder is not repeatable given (obs, noise). Does not count toward budgets."""
    reference = policy.decode(obs, noise)
    for trial in range(CONFIG["determinism_trials"]):
        repeat = policy.decode(obs, noise)
        if not torch.allclose(reference, repeat, atol=CONFIG["determinism_atol"], rtol=0.0):
            max_diff = (reference - repeat).abs().max().item()
            raise RuntimeError(
                f"decoder is not deterministic given noise (trial {trial}, max |diff|={max_diff:.3e}); "
                "DSRL cannot steer it"
            )
    policy.decoder_calls -= obs.shape[0] * (CONFIG["determinism_trials"] + 1)


if __name__ == "__main__":
    class _Linear:
        noise_shape = (2,)

        def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return obs + 0.5 * noise

    class _Stochastic(_Linear):
        def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return obs + noise + 0.01 * torch.randn_like(noise)

    obs_batch, noise_batch = torch.zeros(8, 2), torch.randn(8, 2)
    ok = CountedNoisePolicy(_Linear())
    check_determinism(ok, obs_batch, noise_batch)
    ok.decode(obs_batch, noise_batch)
    print("deterministic decoder accepted; counted calls =", ok.decoder_calls)
    assert ok.decoder_calls == 8

    try:
        check_determinism(CountedNoisePolicy(_Stochastic()), obs_batch, noise_batch)
    except RuntimeError as err:
        print("stochastic decoder rejected:", err)
    else:
        raise AssertionError("stochastic decoder should have been rejected")
