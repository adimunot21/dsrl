"""Diagnostic (not a prereg run): does SAC entropy regularisation explain the bound-limited gap?

Analytic task: decoder a = w, reward = -||a - target||, b_w = 0.8, target (1.5,-1.0) -> optimum -0.728.
Variants differ ONLY in target_entropy. Reports deterministic vs stochastic return and policy spread.
"""
import json, sys, time
import torch
from dsrlstudy import steerer, steerer_batched
from dsrlstudy.noise_policy import CountedNoisePolicy

TARGET = torch.tensor([1.5, -1.0])
B_W = 0.8
ENV_STEPS = 3000  # gap was already at its plateau by 1,600 steps in the plumbing run
SEEDS = [0, 1, 2, 3]

class Identity:
    noise_shape = (2,)
    def decode(self, obs, noise): return noise

sample = lambda n, g: torch.rand(n, 2, generator=g) * 2 - 1
reward = lambda s, a: -(a - TARGET).norm(dim=-1)
optimum = -(TARGET - TARGET.clamp(-B_W, B_W)).norm().item()
TARGETS = [float(x) for x in sys.argv[1:]] or [-2.0, -6.0, -20.0]
out = {"optimum": optimum, "variants": {}}
for target_entropy in TARGETS:
    cfg = dict(steerer.CONFIG)
    cfg.update(total_env_steps=ENV_STEPS, b_w=B_W, device="cuda", cuda_graph=True, log_every=1000, target_entropy=target_entropy)
    policy = CountedNoisePolicy(Identity())
    t0 = time.time()
    results = steerer_batched.train_steerer_ensemble(policy, sample, reward, cfg, SEEDS)
    states = sample(2000, torch.Generator().manual_seed(5))
    det, sto, spread, mean_w = [], [], [], []
    for r in results:
        w_det = r.act(states, deterministic=True).cpu()
        det.append(reward(states, w_det).mean().item())
        draws = torch.stack([r.act(states[:200], deterministic=False).cpu() for _ in range(50)])  # [50,200,2]
        sto.append(reward(states[:200].expand(50, -1, -1), draws).mean().item())
        spread.append(draws.std(dim=0).mean().item())
        mean_w.append(w_det.mean(dim=0).tolist())
    out["variants"][str(target_entropy)] = {"det_return": det, "sto_return": sto, "policy_std": spread,
                                            "mean_deterministic_w": mean_w, "seconds": time.time() - t0}
    print(f"target_entropy={target_entropy}: det {sum(det)/4:.4f} sto {sum(sto)/4:.4f} std {sum(spread)/4:.4f} "
          f"w {mean_w[0]} gap_to_opt {sum(det)/4 - optimum:.4f}", flush=True)
json.dump(out, open(f"results/entropy_bound_test_{'_'.join(map(str, TARGETS))}.json", "w"), indent=1)
print("optimum", optimum)
