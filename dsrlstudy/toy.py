"""Toy contextual task + frozen conditional flow-matching decoder (prereg 001).

Design
------
* State s in [-1,1]^2. K=4 mode centres on a circle of radius R around a state-dependent
  anchor: centre_k(s) = Rot(alpha*s0) @ c_k + shift*s, c_k = R*(cos 2*pi*k/K, sin 2*pi*k/K).
  Mode 0 is the *best* mode.
* Goal g(s) = centre_0(s) + d * u(s), u(s) = Rot(alpha*s0) c_0 / R (radially outward from the
  mode ring, so mode 0 is the nearest data mode at distance exactly d and the others are
  farther). d = support_distance * DATA_STD, DATA_STD = R (= sqrt(trace cov of the centres)).
* Mixture mass: best mode `weakness` (0.5/0.2/0.05), the rest share the remainder equally.
  Each mode is an isotropic Gaussian with std `concentration`.
* Decoder: conditional flow matching (linear path x_t=(1-t)x0+t x1, velocity target x1-x0),
  MLP v(x_t, t, s). Decoding = fixed-step Euler from w in R^2 (w = x0). Deterministic, frozen.
Satisfies `dsrlstudy.noise_policy.NoiseDecoder`.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Dict, List, Tuple

import torch
from torch import nn

from dsrlstudy.ledger import config_hash

logger = logging.getLogger(__name__)

CONFIG = {
    "n_modes": 4,            # K; enough modes for a 'best vs rest' structure, still 2-D visualisable
    "ring_radius": 1.0,      # R; also defines 1 'data std' unit used for support distances
    "rotation_gain": 1.0,    # alpha (rad per unit s0); makes modes state-dependent but smooth
    "shift_gain": 0.5,       # centres drift with s so the task is contextual in both dims
    "support_distances": {"inside": 0.0, "edge": 0.5, "outside": 1.5},  # prereg, in data-std units
    "concentrations": (0.05, 0.15, 0.4),   # prereg mode std
    "weaknesses": (0.5, 0.2, 0.05),        # prereg best-mode mass
    "sparse_radius": 0.1,    # prereg sparse success radius
    "hidden_dim": 256,       # 3x256 MLP: sharp enough for std-0.05 modes, trains in seconds
    "n_hidden_layers": 3,
    "euler_steps": 64,       # fixed decoding steps; std_ratio at 16/32/64/128 steps = 2.17/1.89/1.75/1.69 (converging; residual is model error), 64 is the knee
    "train_steps": 4000,     # chosen after checking mode recovery on the hardest config (std 0.05, best mass 0.05)
    "batch_size": 2048,
    "learning_rate": 2e-3,   # cosine-decayed to 0
    "device": "cuda",        # train AND decode device (falls back to CPU). Probes decode millions of rows; CPU is ~100x too slow on a shared box
}


@dataclass(frozen=True)
class ToyConfig:
    support: str = "inside"        # inside | edge | outside
    concentration: float = 0.15    # mode std
    weakness: float = 0.2          # best-mode mass
    n_modes: int = CONFIG["n_modes"]
    euler_steps: int = CONFIG["euler_steps"]
    train_steps: int = CONFIG["train_steps"]
    batch_size: int = CONFIG["batch_size"]
    learning_rate: float = CONFIG["learning_rate"]
    hidden_dim: int = CONFIG["hidden_dim"]
    n_hidden_layers: int = CONFIG["n_hidden_layers"]

    def __post_init__(self) -> None:
        if self.support not in CONFIG["support_distances"]:
            raise ValueError(f"support must be one of {list(CONFIG['support_distances'])}, got {self.support!r}")
        if self.concentration <= 0:
            raise ValueError("concentration must be > 0")
        if not (0.0 < self.weakness < 1.0):
            raise ValueError("weakness must be in (0,1)")
        if self.n_modes < 2:
            raise ValueError("n_modes must be >= 2")

    @property
    def support_distance(self) -> float:
        return CONFIG["support_distances"][self.support] * CONFIG["ring_radius"]

    @property
    def name(self) -> str:
        return f"{self.support}_c{self.concentration}_w{self.weakness}"

    def hash(self) -> str:
        return config_hash(asdict(self))


def toy_grid() -> List[ToyConfig]:
    """The full 3x3x3 prereg grid (27 configs)."""
    return [ToyConfig(support=s, concentration=c, weakness=w)
            for s, c, w in product(CONFIG["support_distances"], CONFIG["concentrations"], CONFIG["weaknesses"])]


def _rotate(vectors: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """vectors [..., 2] rotated by angles [...]."""
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack([cos * vectors[..., 0] - sin * vectors[..., 1],
                        sin * vectors[..., 0] + cos * vectors[..., 1]], dim=-1)


class ToyTask:
    """Contextual one-step task defined by a ToyConfig. Pure tensor functions, no learned state."""

    obs_dim = 2
    action_dim = 2

    def __init__(self, config: ToyConfig) -> None:
        self.config = config
        k = config.n_modes
        angles = 2 * math.pi * torch.arange(k) / k
        self._ring = CONFIG["ring_radius"] * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [K,2]
        rest = (1.0 - config.weakness) / (k - 1)
        self.mode_weights = torch.tensor([config.weakness] + [rest] * (k - 1))

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Uniform on [-1,1]^2 -> [n,2]."""
        if n <= 0:
            raise ValueError("n must be positive")
        return 2 * torch.rand(n, 2, generator=generator) - 1

    def mode_centres(self, states: torch.Tensor) -> torch.Tensor:
        """[n,2] -> [n,K,2]."""
        self._check_states(states)
        angle = CONFIG["rotation_gain"] * states[:, 0:1].expand(-1, self._ring.shape[0])      # [n,K]
        rotated = _rotate(self._ring.unsqueeze(0).expand(states.shape[0], -1, -1), angle)
        return rotated + CONFIG["shift_gain"] * states.unsqueeze(1)

    def goal(self, states: torch.Tensor) -> torch.Tensor:
        """g(s) = best-mode centre pushed `support_distance` radially outward. [n,2]."""
        centres = self.mode_centres(states)[:, 0]
        ring0 = self._ring[0].expand(states.shape[0], -1)
        outward = _rotate(ring0, CONFIG["rotation_gain"] * states[:, 0]) / CONFIG["ring_radius"]
        return centres + self.config.support_distance * outward

    def reward(self, states: torch.Tensor, actions: torch.Tensor, sparse: bool = False) -> torch.Tensor:
        """Dense: -||a-g||. Sparse: 1[||a-g|| < 0.1]. Returns [n]."""
        self._check_states(states)
        if actions.shape != states.shape:
            raise ValueError(f"actions {tuple(actions.shape)} must match states {tuple(states.shape)}")
        dist = torch.linalg.norm(actions - self.goal(states), dim=-1)
        return (dist < CONFIG["sparse_radius"]).float() if sparse else -dist

    def sample_actions(self, states: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw from the data mixture p(a|s). [n,2]."""
        mode_idx = torch.multinomial(self.mode_weights, states.shape[0], replacement=True, generator=generator)
        centres = self.mode_centres(states)[torch.arange(states.shape[0]), mode_idx]
        return centres + self.config.concentration * torch.randn(states.shape[0], 2, generator=generator)

    @staticmethod
    def _check_states(states: torch.Tensor) -> None:
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError(f"states must be [n,2], got {tuple(states.shape)}")


class _VelocityNet(nn.Module):
    def __init__(self, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2 + 1 + 2, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t, states], dim=-1))


class FlowDecoder(nn.Module):
    """Frozen conditional flow-matching decoder. Implements NoiseDecoder."""

    deterministic = True
    noise_shape: Tuple[int, ...] = (2,)

    def __init__(self, config: ToyConfig, train_seed: int) -> None:
        super().__init__()
        self.config = config
        self.train_seed = train_seed
        self.config_hash = config_hash({"toy": asdict(config), "train_seed": train_seed})
        self.velocity = _VelocityNet(config.hidden_dim, config.n_hidden_layers)

    @property
    def device(self) -> torch.device:
        return next(self.velocity.parameters()).device

    @torch.no_grad()
    def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Fixed-step Euler from noise (t=0) to action (t=1).

        Inputs/outputs live on the caller's device (CPU in CountedNoisePolicy); compute runs on self.device.
        """
        out_device = noise.device
        x, obs = noise.to(self.device), obs.to(self.device)
        steps = self.config.euler_steps
        for i in range(steps):
            t = torch.full((x.shape[0], 1), i / steps, dtype=x.dtype, device=x.device)
            x = x + self.velocity(x, t, obs) / steps
        return x.to(out_device)

    def freeze(self) -> "FlowDecoder":
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_decoder(config: ToyConfig, seed: int) -> FlowDecoder:
    """Train a flow-matching decoder on the K-mode mixture, then freeze it. Seeded and repeatable."""
    task = ToyTask(config)
    start = time.time()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        decoder = FlowDecoder(config, seed)
        optimizer = torch.optim.AdamW(decoder.velocity.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_steps)
        device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
        decoder.to(device).train()
        for step in range(1, config.train_steps + 1):
            states = task.sample_states(config.batch_size, generator)
            target_actions = task.sample_actions(states, generator).to(device)
            noise = torch.randn(config.batch_size, 2, generator=generator).to(device)
            t = torch.rand(config.batch_size, 1, generator=generator).to(device)
            states = states.to(device)
            x_t = (1 - t) * noise + t * target_actions
            loss = ((decoder.velocity(x_t, t, states) - (target_actions - noise)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 1000 == 0 or step == config.train_steps:
                logger.info("%s step %d loss %.4f elapsed %.1fs", config.name, step, loss.item(), time.time() - start)
    decoder.freeze()
    logger.info("trained %s hash=%s params=%d in %.1fs", config.name, decoder.config_hash[:10],
                decoder.n_parameters(), time.time() - start)
    return decoder


@torch.no_grad()
def mode_recovery_stats(decoder: FlowDecoder, task: ToyTask, n_states: int = 20, n_noise: int = 5000,
                        seed: int = 0) -> Dict[str, float]:
    """Compare decoded actions (w~N(0,I)) with the data mixture at fixed states.

    Returns: mass_error (max |decoded mass - target mass| over modes, nearest-centre assignment),
    std_ratio (decoded within-mode std / target std, mean over modes), centre_error (mean distance of
    decoded mode means to true centres, in units of mode std... absolute units).
    Note nearest-centre assignment is unreliable when modes overlap (concentration 0.4 on a unit ring).
    """
    generator = torch.Generator().manual_seed(seed)
    states = task.sample_states(n_states, generator)
    k = task.config.n_modes
    mass_err, std_ratio, centre_err = [], [], []
    for i in range(n_states):
        s = states[i:i + 1].expand(n_noise, -1)
        actions = decoder.decode(s, torch.randn(n_noise, 2, generator=generator))
        centres = task.mode_centres(states[i:i + 1])[0]                       # [K,2]
        assign = torch.cdist(actions, centres).argmin(dim=1)
        for m in range(k):
            members = actions[assign == m]
            frac = members.shape[0] / n_noise
            mass_err.append(abs(frac - task.mode_weights[m].item()))
            if members.shape[0] > 10:
                std_ratio.append(members.std(dim=0).mean().item() / task.config.concentration)
                centre_err.append((members.mean(0) - centres[m]).norm().item())
    return {"max_mass_error": max(mass_err), "mean_mass_error": sum(mass_err) / len(mass_err),
            "std_ratio": sum(std_ratio) / len(std_ratio), "centre_error": sum(centre_err) / len(centre_err)}


@torch.no_grad()
def min_goal_distance(decoder: FlowDecoder, task: ToyTask, n_noise: int = 100_000, n_states: int = 5,
                      seed: int = 0, chunk: int = 100_000) -> Dict[str, float]:
    """Min over n_noise noises of ||decode(s,w) - g(s)||, per state; returns min/mean over states."""
    generator = torch.Generator().manual_seed(seed)
    states = task.sample_states(n_states, generator)
    goals = task.goal(states)
    per_state = []
    for i in range(n_states):
        best = float("inf")
        for start in range(0, n_noise, chunk):
            n = min(chunk, n_noise - start)
            actions = decoder.decode(states[i:i + 1].expand(n, -1), torch.randn(n, 2, generator=generator))
            best = min(best, torch.linalg.norm(actions - goals[i], dim=-1).min().item())
        per_state.append(best)
    return {"min": min(per_state), "mean": sum(per_state) / n_states, "max": max(per_state)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from dsrlstudy.noise_policy import CountedNoisePolicy, check_determinism

    assert len(toy_grid()) == 27 and len({c.hash() for c in toy_grid()}) == 27
    for cfg in [ToyConfig("inside", 0.05, 0.05), ToyConfig("outside", 0.15, 0.2), ToyConfig("edge", 0.4, 0.5),
                ToyConfig("outside", 0.4, 0.5)]:
        t0 = time.time()
        dec = train_decoder(cfg, seed=0)
        task = ToyTask(cfg)
        policy = CountedNoisePolicy(dec)
        gen = torch.Generator().manual_seed(1)
        states = task.sample_states(64, gen)
        check_determinism(policy, states, policy.sample_prior_noise(64, gen))
        assert not any(p.requires_grad for p in dec.parameters()) and not dec.training
        rec = mode_recovery_stats(dec, task)
        dist = min_goal_distance(dec, task)
        print(f"{cfg.name} hash={cfg.hash()[:8]} train+eval={time.time() - t0:.1f}s")
        print("  mode recovery:", {k: round(v, 3) for k, v in rec.items()})
        print("  min dist decoded->goal over 100k noises (5 states):", {k: round(v, 3) for k, v in dist.items()})
