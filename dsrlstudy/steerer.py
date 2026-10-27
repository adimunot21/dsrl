"""DSRL-style latent-noise steerer: SAC whose action is the decoder's input noise.

Design (own implementation of standard SAC, Haarnoja et al. 2018, with twin critics,
tanh-squashed Gaussian actor and automatic entropy tuning):

  state s --actor--> w in [-b_w, b_w]^d --frozen decoder--> action --reward_fn--> r

Data flow per iteration: sample `envs_per_step` states -> actor noise -> ONE decoder call per
state -> reward -> replay -> `utd * envs_per_step` gradient steps. So env_steps counts states
processed, gradient_steps == utd * env_steps, decoder_calls == env_steps (training only).

The task is one-step terminal, so the TD target is r + gamma * (1 - done) * (...) with done=1
(target = reward). `gamma` and target networks are kept so the code is reusable for multi-step
use; with done=1 the target networks do not influence learning.

Failure modes: decoder not deterministic (caller's duty, see noise_policy.check_determinism);
reward_fn returning wrong shape -> ValueError; non-finite loss -> RuntimeError.
"""
from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from dsrlstudy.noise_policy import CountedNoisePolicy

logger = logging.getLogger(__name__)

CONFIG = {
    "b_w": 3.0,               # noise box bound per dim; prereg sweeps {0.5..10}, 3 covers N(0,I) at 3 sigma
    "total_env_steps": 50_000,  # prereg budget
    "utd": 20,                # prereg: gradient steps per env step
    "hidden": 128,            # prereg: 3 x 128 networks
    "n_layers": 3,            # prereg
    "lr": 3e-4,               # standard SAC value; prereg tunes on seeds 100-104 only
    "tau": 0.005,             # standard SAC Polyak rate; prereg tunes on seeds 100-104 only
    "gamma": 0.99,            # irrelevant for one-step terminal tasks; kept for multi-step reuse
    "batch": 256,             # standard SAC minibatch
    "replay_size": 100_000,   # >= total_env_steps so nothing is evicted in the prereg budget
    "learning_starts": 1_000, # warm-up env steps with uniform-in-box noise to seed the replay buffer
    "envs_per_step": 8,       # states per iteration: batching cuts Python overhead 8x; env-step
                              # accounting is unchanged (each state counts as one env step)
    "target_entropy": None,   # None -> -dim(noise) (standard SAC heuristic)
    "init_alpha": 1.0,        # standard SAC start; auto-tuned afterwards
    "log_std_min": -5.0,      # clamp keeps tanh-Gaussian numerically stable
    "log_std_max": 2.0,
    "log_every": 5_000,       # env steps between history records / prints
    "device": "cpu",          # CPU is bitwise-reproducible and fast enough for 128-wide MLPs
}


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, noise_dim: int, b_w: float, cfg: dict) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, 2 * noise_dim, cfg["hidden"], cfg["n_layers"])
        self.b_w, self.noise_dim = float(b_w), noise_dim
        self.log_std_min, self.log_std_max = cfg["log_std_min"], cfg["log_std_max"]

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (w [B,d] in +-b_w, log_prob [B] of w under the squashed density)."""
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        if deterministic:
            return self.b_w * torch.tanh(mean), None
        std = log_std.exp()
        pre_tanh = mean + std * torch.randn_like(mean)
        squashed = torch.tanh(pre_tanh)
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # change of variables for tanh and for the b_w scaling
        log_prob = log_prob - (2 * (math.log(2) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))).sum(-1)
        log_prob = log_prob - self.noise_dim * math.log(self.b_w)
        return self.b_w * squashed, log_prob


class _TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, noise_dim: int, b_w: float, cfg: dict) -> None:
        super().__init__()
        self.b_w = float(b_w)
        self.q1 = _mlp(obs_dim + noise_dim, 1, cfg["hidden"], cfg["n_layers"])
        self.q2 = _mlp(obs_dim + noise_dim, 1, cfg["hidden"], cfg["n_layers"])

    def forward(self, obs: torch.Tensor, w: torch.Tensor):
        x = torch.cat([obs, w / self.b_w], dim=-1)  # normalise noise so scale does not depend on b_w
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass
class SteererResult:
    actor: _Actor
    b_w: float
    history: list = field(default_factory=list)
    env_steps: int = 0
    gradient_steps: int = 0
    decoder_calls: int = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """obs [B,obs_dim] (or [obs_dim]) -> noise [B,*noise_shape] (or [*noise_shape]) within +-b_w."""
        single = obs.ndim == 1
        batch_obs = obs.unsqueeze(0) if single else obs
        was_training = self.actor.training
        self.actor.eval()
        w, _ = self.actor(batch_obs.float(), deterministic=deterministic)
        self.actor.train(was_training)
        return w[0] if single else w


class _Replay:
    def __init__(self, capacity: int, obs_dim: int, noise_dim: int) -> None:
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.obs = torch.zeros(capacity, obs_dim)
        self.w = torch.zeros(capacity, noise_dim)
        self.reward = torch.zeros(capacity)

    def add(self, obs: torch.Tensor, w: torch.Tensor, reward: torch.Tensor) -> None:
        for i in range(obs.shape[0]):  # batches are small; keeps wraparound logic trivial
            self.obs[self.ptr], self.w[self.ptr], self.reward[self.ptr] = obs[i], w[i], reward[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = torch.randint(0, self.size, (batch,))
        return self.obs[idx], self.w[idx], self.reward[idx]


def _validate(cfg: dict) -> None:
    for key in ("b_w", "lr", "tau"):
        if not cfg[key] > 0:
            raise ValueError(f"{key} must be > 0, got {cfg[key]}")
    for key in ("total_env_steps", "utd", "batch", "replay_size", "envs_per_step", "hidden", "n_layers"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {cfg[key]}")
    if cfg["learning_starts"] < 0 or not 0 <= cfg["gamma"] <= 1:
        raise ValueError("learning_starts must be >= 0 and gamma in [0,1]")


def train_steerer(
    policy: CountedNoisePolicy,
    sample_states: Callable[[int, torch.Generator], torch.Tensor],
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: dict,
    seed: int,
    run=None,
) -> SteererResult:
    """Train a SAC steerer on the one-step contextual task. `config` overrides CONFIG keys."""
    cfg = {**CONFIG, **config}
    unknown = set(config) - set(CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    _validate(cfg)
    torch.manual_seed(seed)
    state_gen = torch.Generator().manual_seed(seed + 1_000_003)  # separate stream for start states
    device = torch.device(cfg["device"])

    probe_obs = sample_states(1, state_gen)
    if probe_obs.ndim != 2:
        raise ValueError(f"sample_states must return [n, obs_dim], got {tuple(probe_obs.shape)}")
    obs_dim, noise_shape = probe_obs.shape[1], policy.noise_shape
    noise_dim = int(math.prod(noise_shape))
    b_w = float(cfg["b_w"])

    actor = _Actor(obs_dim, noise_dim, b_w, cfg).to(device)
    critic = _TwinCritic(obs_dim, noise_dim, b_w, cfg).to(device)
    critic_target = copy.deepcopy(critic).requires_grad_(False)
    log_alpha = torch.tensor(math.log(cfg["init_alpha"]), device=device, requires_grad=True)
    target_entropy = -float(noise_dim) if cfg["target_entropy"] is None else float(cfg["target_entropy"])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg["lr"])
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg["lr"])
    replay = _Replay(cfg["replay_size"], obs_dim, noise_dim)
    result = SteererResult(actor=actor, b_w=b_w)

    start = time.perf_counter()
    recent_rewards: list[float] = []
    last = {"critic_loss": float("nan"), "actor_loss": float("nan")}
    next_log = cfg["log_every"]
    calls_before = policy.decoder_calls

    while result.env_steps < cfg["total_env_steps"]:
        n_new = min(cfg["envs_per_step"], cfg["total_env_steps"] - result.env_steps)
        obs = sample_states(n_new, state_gen).float()
        with torch.no_grad():
            if result.env_steps < cfg["learning_starts"]:
                w = (torch.rand(n_new, noise_dim) * 2 - 1) * b_w
            else:
                w, _ = actor(obs.to(device))
                w = w.cpu()
        action = policy.decode(obs, w.reshape(n_new, *noise_shape))  # exactly 1 decoder call / state
        reward = reward_fn(obs, action).float().reshape(-1)
        if reward.shape[0] != n_new:
            raise ValueError(f"reward_fn must return one reward per state, got {tuple(reward.shape)}")
        replay.add(obs, w, reward)
        recent_rewards.extend(reward.tolist())
        result.env_steps += n_new

        if result.env_steps >= cfg["learning_starts"] and replay.size >= cfg["batch"]:
            for _ in range(cfg["utd"] * n_new):
                last = _sac_update(actor, critic, critic_target, log_alpha, target_entropy,
                                   actor_opt, critic_opt, alpha_opt, replay, cfg, device)
                result.gradient_steps += 1

        if result.env_steps >= next_log or result.env_steps >= cfg["total_env_steps"]:
            next_log += cfg["log_every"]
            last = {k: float(v) for k, v in last.items()}
            if not math.isfinite(last["critic_loss"]):
                raise RuntimeError("critic loss became non-finite")
            record = {
                "step": result.env_steps,
                "mean_reward": float(sum(recent_rewards) / max(len(recent_rewards), 1)),
                "critic_loss": last["critic_loss"], "actor_loss": last["actor_loss"],
                "alpha": float(log_alpha.detach().exp()), "elapsed_s": time.perf_counter() - start,
            }
            recent_rewards = []
            result.history.append(record)
            logger.info("%s", record)
            print(f"[steerer] step {record['step']:>7d} | reward {record['mean_reward']:9.4f} | "
                  f"critic {record['critic_loss']:9.4f} | actor {record['actor_loss']:9.4f} | "
                  f"alpha {record['alpha']:.4f} | {record['elapsed_s']:6.1f}s", flush=True)
            if run is not None:
                run.log_metric("steerer_mean_reward", record["mean_reward"], record["step"])

    result.decoder_calls = policy.decoder_calls - calls_before
    result.actor = actor.cpu().eval()
    if run is not None:
        run.add_counts(env_steps=result.env_steps, gradient_steps=result.gradient_steps,
                       decoder_calls=result.decoder_calls)
    return result


def _sac_update(actor, critic, critic_target, log_alpha, target_entropy,
                actor_opt, critic_opt, alpha_opt, replay, cfg, device) -> dict:
    obs, w, reward = (t.to(device) for t in replay.sample(cfg["batch"]))
    # One-step terminal task: done = 1, so target = reward (gamma * (1-done) * ... vanishes).
    # For multi-step reuse, the bootstrap term would use critic_target and the actor on next_obs.
    target = reward
    q1, q2 = critic(obs, w)
    critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    alpha = log_alpha.exp().detach()
    w_new, log_prob = actor(obs)
    q1_pi, q2_pi = critic(obs, w_new)
    actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()

    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    with torch.no_grad():
        for p, p_t in zip(critic.parameters(), critic_target.parameters()):
            p_t.mul_(1 - cfg["tau"]).add_(cfg["tau"] * p)
    # losses stay as detached tensors: .item() every step forces a device sync (slow)
    return {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach()}


if __name__ == "__main__":
    class _Identity:
        noise_shape = (2,)

        def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return noise

    target_point = torch.tensor([1.0, -1.0])
    pol = CountedNoisePolicy(_Identity())
    res = train_steerer(
        pol, lambda n, g: torch.randn(n, 1, generator=g),
        lambda s, a: -(a - target_point).norm(dim=-1),
        {"b_w": 2.0, "total_env_steps": 3000, "log_every": 1000, "learning_starts": 200, "utd": 5}, seed=0)
    out = res.act(torch.zeros(1, 1))
    print("learned noise", out, "target", target_point)
    assert res.decoder_calls == res.env_steps == 3000
