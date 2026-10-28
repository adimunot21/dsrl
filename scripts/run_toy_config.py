"""Run one toy config under prereg 001 / amendments 001a-b and write results.json.

Modes:
  tune   : train a 5-seed ensemble at reduced budget for one learning rate (seeds 100-104)
  main   : train a 10-seed ensemble (seeds 0-9) at the full 001 budget, then evaluate + probe
  bound  : analytic bound-limited steerer check (linear decoder), full budget
Every mode goes through RunLedger and writes runs/toy/<run_id>/results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from dsrlstudy import probe, steerer, steerer_batched, toy
from dsrlstudy.ledger import RunLedger
from dsrlstudy.noise_policy import CountedNoisePolicy, check_determinism

logger = logging.getLogger("run_toy_config")

CONFIG = {
    "main_seeds": list(range(10)),          # 001: report on seeds 0-9
    "tune_seeds": list(range(100, 105)),    # 001/001b: tuning seeds, never reported
    "tune_env_steps": 8_000,                # 001b: reduced tuning budget
    "full_env_steps": 50_000,               # 001: matched budget
    "b_w": 1.5,                             # 001b: main-grid noise bound
    "decoder_seed": 0,                      # 001b: one frozen decoder per config
    "eval_states": 2_000,                   # 001b
    "ref_states": 100,                      # 001b: C_ref states
    "ref_noise": 100_000,                   # 001: C_ref
    "ceiling_ns": [16, 64, 256, 1024],      # 001: reported N
    "eval_stream_seed": 777,                # held-out evaluation stream
    "probe_stream_seed": 778,               # probe stream, separate from evaluation
}


def _git_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def run_main(args, run_id: str, out_dir: Path) -> dict:
    cfg = toy.ToyConfig(support=args.support, concentration=args.concentration, weakness=args.weakness)
    task = toy.ToyTask(cfg)
    t0 = time.time()
    decoder = toy.train_decoder(cfg, seed=CONFIG["decoder_seed"])
    policy = CountedNoisePolicy(decoder)
    probe_gen = torch.Generator().manual_seed(CONFIG["probe_stream_seed"])
    check_determinism(policy, task.sample_states(64, probe_gen), torch.randn(64, 2, generator=probe_gen))
    logger.info("decoder ready in %.0fs", time.time() - t0)

    gap = toy.min_goal_distance(decoder, task)                       # 001a reachability gap
    j0 = probe.base_return(policy, task, CONFIG["eval_states"], torch.Generator().manual_seed(CONFIG["eval_stream_seed"]))
    probes = {"reachability_gap": gap, "J0": j0.value}
    if args.mode == "main":  # tuning does not use the probes (001b); skip ~4 min
        ceilings = probe.search_ceiling_multi(policy, task, CONFIG["ceiling_ns"], CONFIG["eval_states"],
                                              torch.Generator().manual_seed(CONFIG["probe_stream_seed"]))
        c_ref = probe.search_ceiling(policy, task, CONFIG["ref_noise"], CONFIG["ref_states"],
                                     torch.Generator().manual_seed(CONFIG["probe_stream_seed"] + 1))
        box = probe.search_ceiling_multi(policy, task, CONFIG["ceiling_ns"], CONFIG["eval_states"],
                                         torch.Generator().manual_seed(CONFIG["probe_stream_seed"]), box_bound=CONFIG["b_w"])
        ent = probe.base_action_entropy(policy, task, CONFIG["eval_states"], torch.Generator().manual_seed(CONFIG["probe_stream_seed"] + 2))
        var = probe.action_variance(policy, task, CONFIG["eval_states"], torch.Generator().manual_seed(CONFIG["probe_stream_seed"] + 3))
        probes.update({
            "reachability_gap": gap, "J0": j0.value,
            "C": {str(n): {"value": r.value, "decoder_calls": r.decoder_calls} for n, r in ceilings.items()},
            "C_ref": {"value": c_ref.value, "decoder_calls": c_ref.decoder_calls, "n_states": CONFIG["ref_states"]},
            "C_box": {str(n): {"value": r.value, "decoder_calls": r.decoder_calls} for n, r in box.items()},
            "H_probe_box_256": probe.headroom(box[256].value, j0.value),
            "H_ent": ent.value, "action_variance": var.value,
            "H_probe_256": probe.headroom(ceilings[256].value, j0.value),
    })
    logger.info("probes: %s", json.dumps(probes)[:400])

    seeds = CONFIG["main_seeds"] if args.mode == "main" else CONFIG["tune_seeds"]
    env_steps = CONFIG["full_env_steps"] if args.mode == "main" else CONFIG["tune_env_steps"]
    steer_cfg = dict(steerer.CONFIG)
    steer_cfg.update(total_env_steps=env_steps, b_w=CONFIG["b_w"], lr=args.lr, device="cuda",
                     cuda_graph=True, log_every=max(env_steps // 10, 1), target_entropy=args.target_entropy)
    policy.reset_counter()
    with RunLedger("runs/ledger", run_id, {**vars(args), **CONFIG, "steerer": steer_cfg, "toy": cfg.hash()},
                   seed=seeds[0], label=f"toy-{args.mode}", num_gpus=1, cpu_cores_used=2,
                   repo_path=_git_root()) as run:
        results = steerer_batched.train_steerer_ensemble(
            policy, lambda n, g: task.sample_states(n, g), lambda s, a: task.reward(s, a), steer_cfg, seeds, run=run)
        eval_states = task.sample_states(CONFIG["eval_states"], torch.Generator().manual_seed(CONFIG["eval_stream_seed"] + 1))
        per_seed = []
        for res in results:
            noise = res.act(eval_states, deterministic=True).cpu()
            per_seed.append(task.reward(eval_states, policy.decode(eval_states, noise)).mean().item())
    gains = [r - j0.value for r in per_seed]
    return {"toy_name": cfg.name, "toy_hash": cfg.hash(), "probes": probes, "seeds": seeds,
            "per_seed_return": per_seed, "per_seed_gain": gains, "env_steps": env_steps,
            "lr": args.lr, "b_w": CONFIG["b_w"],
            "histories": [r.history for r in results]}


def run_bound(args, run_id: str, out_dir: Path) -> dict:
    """Analytic check: decoder a = w, reward = -||a - target||, bound b_w=0.8 -> optimum -(||target|| - 0.8*sqrt(2)... computed below."""
    target = torch.tensor([1.5, -1.0])
    b_w = 0.8
    class Identity:
        noise_shape = (2,)
        def decode(self, obs, noise): return noise
    policy = CountedNoisePolicy(Identity())
    # optimum over the box [-b,b]^2: clip target into the box, distance to it
    optimum = -(target - target.clamp(-b_w, b_w)).norm().item()
    steer_cfg = dict(steerer.CONFIG)
    steer_cfg.update(total_env_steps=CONFIG["full_env_steps"], b_w=b_w, lr=args.lr, device="cuda",
                     cuda_graph=True, log_every=5_000)
    seeds = [0, 1, 2, 3]
    sample = lambda n, g: torch.rand(n, 2, generator=g) * 2 - 1
    reward = lambda s, a: -(a - target).norm(dim=-1)
    with RunLedger("runs/ledger", run_id, {**vars(args), **CONFIG, "steerer": steer_cfg}, seed=0,
                   label="toy-bound-check", num_gpus=1, cpu_cores_used=2, repo_path=_git_root()) as run:
        results = steerer_batched.train_steerer_ensemble(policy, sample, reward, steer_cfg, seeds, run=run)
        states = sample(2000, torch.Generator().manual_seed(5))
        per_seed = [reward(states, policy.decode(states, r.act(states, deterministic=True).cpu())).mean().item() for r in results]
    return {"analytic_optimum": optimum, "per_seed_return": per_seed,
            "gap_to_optimum": [r - optimum for r in per_seed], "max_exceeds_bound": max(per_seed) > optimum + 1e-3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "main", "bound"], required=True)
    parser.add_argument("--support", default="inside")
    parser.add_argument("--concentration", type=float, default=0.15)
    parser.add_argument("--weakness", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-entropy", type=float, default=None,
                        help="SAC target entropy; default None = -dim (session 1). DSRL's own setting is 0.0 (amendment 001c)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--env-steps", type=int, default=0, help="override env-step budget (plumbing tests only)")
    args = parser.parse_args()
    if args.env_steps:
        CONFIG["full_env_steps"] = CONFIG["tune_env_steps"] = args.env_steps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_id = f"toy_{args.mode}_{args.support}_{args.concentration}_{args.weakness}_lr{args.lr}_ent{args.target_entropy}{args.tag}"
    out_dir = Path("runs/toy") / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    payload = run_bound(args, run_id, out_dir) if args.mode == "bound" else run_main(args, run_id, out_dir)
    payload["wall_clock_s"] = time.time() - start
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: v for k, v in payload.items() if k != "histories"}, indent=1)[:2500])


if __name__ == "__main__":
    main()
