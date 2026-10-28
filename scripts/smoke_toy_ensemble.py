import sys, time, torch, numpy as np
sys.path.insert(0, "/home/adimunot21/dev/dsrl")
from dsrlstudy import toy, probe, steerer, steerer_batched
from dsrlstudy.noise_policy import CountedNoisePolicy, check_determinism
t0 = time.time()
cfg = toy.ToyConfig(support="inside", concentration=0.15, weakness=0.2)
task = toy.ToyTask(cfg)
dec = toy.train_decoder(cfg, seed=900)
print(f"[1] decoder trained {time.time()-t0:.0f}s, hash {cfg.hash()[:8]}")
pol = CountedNoisePolicy(dec)
g = torch.Generator().manual_seed(1)
s = task.sample_states(64, g); check_determinism(pol, s, torch.randn(64, 2, generator=g)); print("[2] determinism ok")
ev = torch.Generator().manual_seed(777)
j0 = probe.base_return(pol, task, 2000, ev)
cs = probe.search_ceiling_multi(pol, task, [16, 256], 500, torch.Generator().manual_seed(778))
print(f"[3] J0={j0.value:.3f} ({j0.decoder_calls} calls) C16={cs[16].value:.3f} C256={cs[256].value:.3f} ({cs[256].decoder_calls} calls) H_probe={probe.headroom(cs[256].value, j0.value):.3f}")
pol.reset_counter()
scfg = dict(steerer.CONFIG); scfg.update(total_env_steps=5000, b_w=1.5, device="cuda", cuda_graph=True, log_every=2500)
t1 = time.time()
res = steerer_batched.train_steerer_ensemble(pol, lambda n, gen: task.sample_states(n, gen), lambda st, a: task.reward(st, a), scfg, seeds=list(range(900, 910)))
dt = time.time() - t1
print(f"[4] ensemble S=10 trained {dt:.0f}s; env_steps={res[0].env_steps} grad={res[0].gradient_steps} decoder_calls total={pol.decoder_calls}")
es = task.sample_states(2000, torch.Generator().manual_seed(779))
rets = []
for r in res:
    w = r.act(es, deterministic=True).cpu()
    assert w.abs().max() <= 1.5 + 1e-4
    rets.append(task.reward(es, pol.decode(es, w)).mean().item())
print("[5] steered return per seed:", np.round(rets, 3))
print(f"    mean G = {np.mean(rets)-j0.value:.3f}  vs probe headroom {probe.headroom(cs[256].value, j0.value):.3f}  total {time.time()-t0:.0f}s")
