"""Ensemble SAC steerer: S independent steerers (one per seed) trained in ONE batched pass.

Why: steerer.py costs ~50 ms/gradient step here because per-op Python overhead dominates, not FLOPs.
Stacking S members' parameters into [S, in, out] tensors and using torch.baddbmm makes one op serve all
members, so the overhead is amortised S-fold.

Design / contracts
  * Members share NOTHING: parameters, replay buffers, entropy temperatures (log_alpha [S]), RNG
    generators (one torch.Generator per member, seeded from its seed) are all per member.
  * Single optimiser step over stacked tensors == S independent Adams, because (a) Adam is elementwise
    and (b) the objective we differentiate is the SUM (not mean) of per-member losses, so the gradient
    that reaches member m's slice equals the gradient of member m's own loss (member losses are
    additive and depend only on that member's slice). A mean over members would scale every gradient
    by 1/S. Adam is nearly scale-invariant but not exactly (eps), so we use the sum. Tested.
  * LayerNorm is manual over the last dim with per-member affine params (eps=1e-5, biased variance =
    nn.LayerNorm semantics).
  * Initial weights are produced by building the reference steerer.py _Actor/_TwinCritic modules under
    each member's seed and stacking them, so inits match train_steerer(seed) exactly.
  * Twin critics are stacked as 2S "members" (rows 0..S-1 = q1, S..2S-1 = q2).
  * One decoder call per member per state: the S*n states of an iteration go through ONE batched
    policy.decode (row-counted), so policy.decoder_calls grows by S*n per iteration; per-member
    decoder_calls == env_steps is asserted. The decoder must be row-wise (rows independent), as for
    any frozen noise->action map.
  * reward_fn(obs, action) keeps the single-run signature and is called once per member per
    iteration, in member order 0..S-1.
  * Random streams of a member do not depend on S or on the other members (gradient-step randomness
    is pre-drawn in fixed-size chunks that depend only on batch and noise_dim).

Not bitwise equal to steerer.py (different RNG streams, bmm summation order): equivalence is
statistical. Failure modes: bad reward shape -> ValueError; non-finite loss -> RuntimeError.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from dsrlstudy.noise_policy import CountedNoisePolicy
from dsrlstudy.steerer import CONFIG as SINGLE_CONFIG
from dsrlstudy.steerer import SteererResult, _Actor, _TwinCritic, _validate

logger = logging.getLogger(__name__)

CONFIG = {
    **SINGLE_CONFIG,  # same keys/semantics as steerer.py
    "device": "cpu",         # benchmark (see __main__ / report): choose per measured ms/member-step
    "cuda_graph": False,     # capture the whole update in a CUDA graph (cuda only): removes launch overhead
    "compile": False,        # torch.compile the update step; off by default (see benchmark notes)
    "rng_chunk_elems": 1 << 20,  # cap on pre-drawn actor-noise elements per member per chunk (memory bound)
    "layernorm_eps": 1e-5,   # nn.LayerNorm default, needed for equivalence with steerer.py
}


def _disable_native_triton_overrides() -> None:
    """torch 2.14 routes some CUDA bmm backward shapes to Triton kernels that must be JIT-compiled with
    cuda.h; this machine has no CUDA headers, so that path crashes. Fall back to cuBLAS. Best effort."""
    try:
        from torch._native import registry
        registry.deregister_op_overrides(disable_dsl_names="triton")
    except Exception as err:  # private API; absence just means nothing to disable
        logger.debug("could not disable native triton overrides: %s", err)


class _BatchedMLP(nn.Module):
    """N independent MLPs (Linear, LayerNorm, ReLU)*L + Linear evaluated with baddbmm. x: [N,B,in]."""

    def __init__(self, members: list[nn.Sequential], eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weights = nn.ParameterList()   # per layer [N, in, out]
        self.biases = nn.ParameterList()    # per layer [N, 1, out]
        self.gammas = nn.ParameterList()    # per hidden layer [N, 1, h]
        self.betas = nn.ParameterList()
        linears = [[m for m in seq if isinstance(m, nn.Linear)] for seq in members]
        norms = [[m for m in seq if isinstance(m, nn.LayerNorm)] for seq in members]
        for layer in range(len(linears[0])):
            self.weights.append(nn.Parameter(torch.stack([l[layer].weight.detach().t() for l in linears]).contiguous()))
            self.biases.append(nn.Parameter(torch.stack([l[layer].bias.detach() for l in linears]).unsqueeze(1).contiguous()))
        for layer in range(len(norms[0])):
            self.gammas.append(nn.Parameter(torch.stack([n[layer].weight.detach() for n in norms]).unsqueeze(1).contiguous()))
            self.betas.append(nn.Parameter(torch.stack([n[layer].bias.detach() for n in norms]).unsqueeze(1).contiguous()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_hidden = len(self.gammas)
        for layer in range(n_hidden):
            x = torch.baddbmm(self.biases[layer], x, self.weights[layer])
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            x = torch.relu((x - mean) * torch.rsqrt(var + self.eps) * self.gammas[layer] + self.betas[layer])
        return torch.baddbmm(self.biases[n_hidden], x, self.weights[n_hidden])

    @torch.no_grad()
    def write_member(self, index: int, seq: nn.Sequential) -> None:
        """Copy member `index`'s parameters into a reference nn.Sequential MLP."""
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        norms = [m for m in seq if isinstance(m, nn.LayerNorm)]
        for layer, lin in enumerate(linears):
            lin.weight.copy_(self.weights[layer][index].t())
            lin.bias.copy_(self.biases[layer][index, 0])
        for layer, norm in enumerate(norms):
            norm.weight.copy_(self.gammas[layer][index, 0])
            norm.bias.copy_(self.betas[layer][index, 0])


class _Ensemble:
    """All state of S independent steerers: networks, optimisers, replay, RNGs."""

    def __init__(self, seeds: list[int], obs_dim: int, noise_dim: int, cfg: dict, device: torch.device) -> None:
        if device.type == "cuda":
            _disable_native_triton_overrides()
        self.seeds, self.S, self.cfg, self.device = list(seeds), len(seeds), cfg, device
        self.obs_dim, self.noise_dim, self.b_w = obs_dim, noise_dim, float(cfg["b_w"])
        actors, critics = [], []
        for seed in seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)  # same construction order as train_steerer -> identical init
                actors.append(_Actor(obs_dim, noise_dim, self.b_w, cfg))
                critics.append(_TwinCritic(obs_dim, noise_dim, self.b_w, cfg))
        eps = cfg["layernorm_eps"]
        self.actor = _BatchedMLP([a.net for a in actors], eps).to(device)
        self.critic = _BatchedMLP([c.q1 for c in critics] + [c.q2 for c in critics], eps).to(device)
        self.critic_target = _BatchedMLP([c.q1 for c in critics] + [c.q2 for c in critics], eps).to(device)
        self.critic_target.requires_grad_(False)
        self.log_alpha = torch.full((self.S,), math.log(cfg["init_alpha"]), device=device, requires_grad=True)
        self.target_entropy = -float(noise_dim) if cfg["target_entropy"] is None else float(cfg["target_entropy"])
        fused = device.type == "cuda"
        graph = bool(cfg["cuda_graph"]) and fused
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"], fused=fused, capturable=graph)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"], fused=fused, capturable=graph)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["lr"], fused=fused, capturable=graph)
        self.actor_params = list(self.actor.parameters())
        self.rngs = [torch.Generator(device=device).manual_seed(s + 2_000_003) for s in seeds]
        cap = cfg["replay_size"]
        self.buf_obs = torch.zeros(self.S, cap, obs_dim, device=device)
        self.buf_w = torch.zeros(self.S, cap, noise_dim, device=device)
        self.buf_r = torch.zeros(self.S, cap, device=device)
        self.ptr, self.size = 0, 0
        self.member_rows = torch.arange(self.S, device=device).unsqueeze(1)
        self.chunk_steps = max(1, min(64, cfg["rng_chunk_elems"] // (cfg["batch"] * max(noise_dim, 1))))
        self._idx = self._eps = None
        self._chunk_pos = self.chunk_steps  # force refill
        update = self._update
        self._update_fn = torch.compile(update) if cfg["compile"] else update
        self._use_graph, self._graph, self._g_idx, self._g_eps, self._g_out = graph, None, None, None, None

    # ---- policy -------------------------------------------------------------------------------
    def _heads(self, obs: torch.Tensor):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.cfg["log_std_min"], self.cfg["log_std_max"])

    @torch.no_grad()
    def act_stochastic(self, obs: torch.Tensor, n: int) -> torch.Tensor:
        mean, log_std = self._heads(obs)
        eps = torch.stack([torch.randn(n, self.noise_dim, generator=g, device=self.device) for g in self.rngs])
        return self.b_w * torch.tanh(mean + log_std.exp() * eps)

    @torch.no_grad()
    def uniform_noise(self, n: int) -> torch.Tensor:
        u = torch.stack([torch.rand(n, self.noise_dim, generator=g, device=self.device) for g in self.rngs])
        return (u * 2 - 1) * self.b_w

    # ---- replay -------------------------------------------------------------------------------
    @torch.no_grad()
    def add(self, obs: torch.Tensor, w: torch.Tensor, reward: torch.Tensor) -> None:
        """obs [S,n,od], w [S,n,d], reward [S,n]; ring buffer with a pointer shared by all members
        (members see identical env-step counts, but each owns its rows)."""
        n = obs.shape[1]
        rows = (self.ptr + torch.arange(n, device=self.device)) % self.cfg["replay_size"]
        self.buf_obs[:, rows], self.buf_w[:, rows], self.buf_r[:, rows] = obs, w, reward
        self.ptr = (self.ptr + n) % self.cfg["replay_size"]
        self.size = min(self.size + n, self.cfg["replay_size"])

    @torch.no_grad()
    def _refill(self) -> None:
        k, B = self.chunk_steps, self.cfg["batch"]
        self._idx = torch.stack([torch.randint(0, self.size, (k, B), generator=g, device=self.device) for g in self.rngs])
        self._eps = torch.stack([torch.randn(k, B, self.noise_dim, generator=g, device=self.device) for g in self.rngs])
        self._chunk_pos = 0

    # ---- one SAC step for all members ----------------------------------------------------------
    def _update(self, idx: torch.Tensor, eps: torch.Tensor):
        S, b_w = self.S, self.b_w
        obs, w, reward = self.buf_obs[self.member_rows, idx], self.buf_w[self.member_rows, idx], self.buf_r[self.member_rows, idx]
        # critic: one-step terminal task => target = reward (as in steerer.py)
        q = self.critic(torch.cat([obs, w / b_w], -1).repeat(2, 1, 1)).squeeze(-1)  # [2S,B]
        target = reward.repeat(2, 1)
        critic_loss_m = ((q - target) ** 2).mean(1).view(2, S).sum(0)  # per member: mse(q1)+mse(q2)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_m.sum().backward()  # SUM over members: see module docstring
        self.critic_opt.step()

        alpha = self.log_alpha.detach().exp().unsqueeze(1)  # [S,1]
        mean, log_std = self._heads(obs)
        std = log_std.exp()
        pre_tanh = mean + std * eps
        w_new = b_w * torch.tanh(pre_tanh)
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))).sum(-1)
        log_prob = log_prob - self.noise_dim * math.log(b_w)
        q_pi = self.critic(torch.cat([obs, w_new / b_w], -1).repeat(2, 1, 1)).squeeze(-1)
        actor_loss_m = (alpha * log_prob - torch.min(q_pi[:S], q_pi[S:])).mean(1)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_m.sum().backward(inputs=self.actor_params)  # critic params not touched by this backward
        self.actor_opt.step()

        alpha_loss_m = -(self.log_alpha.unsqueeze(1) * (log_prob.detach() + self.target_entropy)).mean(1)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss_m.sum().backward()
        self.alpha_opt.step()

        with torch.no_grad():
            torch._foreach_lerp_(list(self.critic_target.parameters()), list(self.critic.parameters()), self.cfg["tau"])
        return critic_loss_m.detach(), actor_loss_m.detach()

    def step(self):
        if self._chunk_pos >= self.chunk_steps:
            self._refill()
        i = self._chunk_pos
        self._chunk_pos += 1
        if not self._use_graph:
            return self._update_fn(self._idx[:, i], self._eps[:, i])
        if self._graph is None:
            self._capture(self._idx[:, i], self._eps[:, i])
        self._g_idx.copy_(self._idx[:, i])
        self._g_eps.copy_(self._eps[:, i])
        self._graph.replay()
        return self._g_out

    def _capture(self, idx: torch.Tensor, eps: torch.Tensor) -> None:
        """Capture one update in a CUDA graph. Warm-up steps run on a side stream first (they are REAL
        updates on the first sampled batch: 3 extra gradient steps are not counted, so graph mode
        is applied only after the caller's first step; we undo nothing, so we instead run warm-up on
        copies of the state and restore it)."""
        saved = {k: v.clone() for k, v in self._state_tensors().items()}
        self._g_idx, self._g_eps = idx.clone(), eps.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._update(self._g_idx, self._g_eps)
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._g_out = self._update(self._g_idx, self._g_eps)
        # warm-up and capture dirtied parameters/optimiser state: restore the pre-capture snapshot
        with torch.no_grad():
            for k, v in self._state_tensors().items():
                v.copy_(saved[k]) if k in saved else v.zero_()  # fresh Adam state = zeros (step 0)

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, mod in (("actor", self.actor), ("critic", self.critic), ("target", self.critic_target)):
            for k, v in mod.state_dict().items():
                out[f"{name}.{k}"] = v
        out["log_alpha"] = self.log_alpha.data
        for name, opt in (("a", self.actor_opt), ("c", self.critic_opt), ("l", self.alpha_opt)):
            for j, st in enumerate(opt.state.values()):
                for k, v in st.items():
                    out[f"opt{name}{j}.{k}"] = v
        return out

    # ---- export -------------------------------------------------------------------------------
    def export(self, member: int, obs_dim: int, noise_dim: int) -> _Actor:
        actor = _Actor(obs_dim, noise_dim, self.b_w, self.cfg)
        self.actor.write_member(member, actor.net)
        return actor.cpu().eval()


def train_steerer_ensemble(
    policy: CountedNoisePolicy,
    sample_states: Callable[[int, torch.Generator], torch.Tensor],
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: dict,
    seeds: list[int],
    run=None,
) -> list[SteererResult]:
    """Train len(seeds) independent SAC steerers at once. Returns one SteererResult per seed.

    `config` overrides CONFIG keys. `run` (optional) gets per-seed metrics `steerer_mean_reward_seed<k>`
    and add_counts summed over members.
    """
    cfg = {**CONFIG, **config}
    unknown = set(config) - set(CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    _validate(cfg)
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty list of distinct ints, got {seeds}")
    device = torch.device(cfg["device"])
    S = len(seeds)
    state_gens = [torch.Generator().manual_seed(s + 1_000_003) for s in seeds]  # same stream as steerer.py

    probe = [sample_states(1, g) for g in state_gens]
    if probe[0].ndim != 2:
        raise ValueError(f"sample_states must return [n, obs_dim], got {tuple(probe[0].shape)}")
    obs_dim, noise_shape = probe[0].shape[1], policy.noise_shape
    noise_dim = int(math.prod(noise_shape))
    ens = _Ensemble(seeds, obs_dim, noise_dim, cfg, device)
    results = [SteererResult(actor=None, b_w=ens.b_w) for _ in seeds]  # type: ignore[arg-type]

    start = time.perf_counter()
    env_steps = grad_steps = 0
    recent: list[list[float]] = [[] for _ in seeds]
    last = (torch.full((S,), float("nan")), torch.full((S,), float("nan")))
    next_log = cfg["log_every"]
    calls_before = policy.decoder_calls

    while env_steps < cfg["total_env_steps"]:
        n_new = min(cfg["envs_per_step"], cfg["total_env_steps"] - env_steps)
        obs = torch.stack([sample_states(n_new, g).float() for g in state_gens])  # [S,n,od] (CPU)
        obs_dev = obs.to(device)
        if env_steps < cfg["learning_starts"]:
            w = ens.uniform_noise(n_new)
        else:
            w = ens.act_stochastic(obs_dev, n_new)
        w_cpu = w.cpu()
        # exactly one decoder call per member per state (S*n rows in one batched call)
        action = policy.decode(obs.reshape(S * n_new, obs_dim), w_cpu.reshape(S * n_new, *noise_shape))
        action = action.reshape(S, n_new, *action.shape[1:])
        reward = torch.stack([reward_fn(obs[m], action[m]).float().reshape(-1) for m in range(S)])
        if reward.shape != (S, n_new):
            raise ValueError(f"reward_fn must return one reward per state, got per-member {tuple(reward.shape[1:])}")
        ens.add(obs_dev, w, reward.to(device))
        for m in range(S):
            recent[m].extend(reward[m].tolist())
        env_steps += n_new

        if env_steps >= cfg["learning_starts"] and ens.size >= cfg["batch"]:
            for _ in range(cfg["utd"] * n_new):
                last = ens.step()
                grad_steps += 1

        if env_steps >= next_log or env_steps >= cfg["total_env_steps"]:
            next_log += cfg["log_every"]
            critic_loss, actor_loss = last[0].cpu(), last[1].cpu()
            if not bool(torch.isfinite(critic_loss).all()):
                raise RuntimeError(f"critic loss became non-finite for members {torch.nonzero(~torch.isfinite(critic_loss)).flatten().tolist()}")
            alpha = ens.log_alpha.detach().exp().cpu()
            elapsed = time.perf_counter() - start
            for m, res in enumerate(results):
                record = {"step": env_steps, "mean_reward": float(sum(recent[m]) / max(len(recent[m]), 1)),
                          "critic_loss": float(critic_loss[m]), "actor_loss": float(actor_loss[m]),
                          "alpha": float(alpha[m]), "elapsed_s": elapsed}
                res.history.append(record)
                recent[m] = []
                if run is not None:
                    run.log_metric(f"steerer_mean_reward_seed{seeds[m]}", record["mean_reward"], env_steps)
            logger.info("ensemble step %d rewards %s", env_steps, [round(r.history[-1]["mean_reward"], 4) for r in results])
            print(f"[ensemble S={S}] step {env_steps:>7d} | reward mean {sum(r.history[-1]['mean_reward'] for r in results) / S:9.4f} "
                  f"| critic {float(critic_loss.mean()):9.4f} | alpha {float(alpha.mean()):.4f} | {elapsed:6.1f}s", flush=True)

    total_calls = policy.decoder_calls - calls_before
    if total_calls != S * env_steps:
        raise RuntimeError(f"decoder calls {total_calls} != members*env_steps {S * env_steps}")
    for m, res in enumerate(results):
        res.env_steps, res.gradient_steps, res.decoder_calls = env_steps, grad_steps, total_calls // S
        res.actor = ens.export(m, obs_dim, noise_dim)
    if run is not None:
        run.add_counts(env_steps=env_steps * S, gradient_steps=grad_steps * S, decoder_calls=total_calls)
    return results


def benchmark_ms_per_step(S: int, device: str, n_steps: int = 200, warmup: int = 30, compile_: bool = False,
                          cuda_graph: bool = False,
                          obs_dim: int = 4, noise_dim: int = 8) -> float:
    """Wall-clock ms per ensemble gradient step (hidden/layers/batch from CONFIG: 128/3/256)."""
    cfg = {**CONFIG, "device": device, "compile": compile_, "cuda_graph": cuda_graph, "replay_size": 4096}
    ens = _Ensemble(list(range(S)), obs_dim, noise_dim, cfg, torch.device(device))
    n = 2048
    ens.add(torch.randn(S, n, obs_dim, device=device), torch.randn(S, n, noise_dim, device=device).clamp(-3, 3),
            torch.randn(S, n, device=device))
    for _ in range(warmup):
        ens.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        ens.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1e3


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _Identity:
        noise_shape = (2,)

        def decode(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return noise

    target_point = torch.tensor([1.0, -1.0])
    pol = CountedNoisePolicy(_Identity())
    out = train_steerer_ensemble(
        pol, lambda n, g: torch.randn(n, 1, generator=g), lambda s, a: -(a - target_point).norm(dim=-1),
        {"b_w": 2.0, "total_env_steps": 3000, "log_every": 1000, "learning_starts": 200, "utd": 5}, seeds=[0, 1, 2])
    for res in out:
        print("learned noise", res.act(torch.zeros(1, 1)), "target", target_point)
        assert res.decoder_calls == res.env_steps == 3000
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for S in (1, 4, 10):
            ms = benchmark_ms_per_step(S, dev)
            print(f"{dev} S={S}: {ms:.2f} ms/step, {ms / S:.2f} ms/member-step")
