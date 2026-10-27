"""Raw env smoke + throughput. Uses DUMMY normalization (real npz is in the Drive checkpoints). No training."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/adimunot21/dev/dsrl/third_party/dsrl"); sys.path.append("/home/adimunot21/dev/dsrl/third_party/dsrl/dppo")
os.chdir("/home/adimunot21/dev/dsrl/third_party/dsrl")
import gym, d4rl, d4rl.gym_mujoco
from omegaconf import OmegaConf
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_utils import ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
SP = os.path.dirname(os.path.abspath(__file__))
def dummy_norm(odim, adim, path):
    np.savez(path, obs_min=-np.ones(odim), obs_max=np.ones(odim), action_min=-np.ones(adim), action_max=np.ones(adim))
    return path
def make_factory(name):
    if name == "can":
        cfg = OmegaConf.create(dict(act_steps=4, obs_dim=23)); norm = dummy_norm(23, 7, f"{SP}/norm_can.npz"); T = 300
        keys = ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']
        def f():
            e = make_robomimic_env(env="can", normalization_path=norm, low_dim_keys=keys, dppo_path="./dppo")
            e = ObservationWrapperRobomimic(e, reward_offset=1)
            return ActionChunkWrapper(e, cfg, max_episode_steps=T)
    else:
        cfg = OmegaConf.create(dict(act_steps=4, obs_dim=11)); norm = dummy_norm(11, 3, f"{SP}/norm_hop.npz"); T = 1000
        def f():
            e = gym.make("hopper-medium-v2"); e = ObservationWrapperGym(e, norm)
            return ActionChunkWrapper(e, cfg, max_episode_steps=T)
    return f
def run(name, n_envs, steps):
    env = make_vec_env(make_factory(name), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env.seed(2); obs = env.reset()
    if n_envs == 1 or True:
        print(f"[{name} n={n_envs}] obs {obs.shape} {obs.dtype} range[{obs.min():.2f},{obs.max():.2f}] act_space {env.action_space.shape} [{env.action_space.low.min()},{env.action_space.high.max()}]")
    rewards = []; eps = 0
    t0 = time.time()
    for _ in range(steps):
        a = np.stack([env.action_space.sample() for _ in range(n_envs)])
        obs, r, d, info = env.step(a); rewards.append(r); eps += int(d.sum())
    dt = time.time() - t0
    rs = np.concatenate(rewards)
    tot = steps * n_envs
    print(f"[{name} n={n_envs}] {tot} chunk-steps in {dt:.1f}s = {tot/dt:.0f} chunk-steps/s = {tot*4/dt:.0f} sim-steps/s; reward range [{rs.min():.3f},{rs.max():.3f}], episodes ended {eps}", flush=True)
    env.close()
if __name__ == "__main__":
    which = sys.argv[1]
    for n in [int(x) for x in sys.argv[2].split(",")]:
        run(which, n, int(sys.argv[3]))
