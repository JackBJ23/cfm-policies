import gymnasium as gym
import numpy as np
from collections import defaultdict
from mani_skill.utils.wrappers import CPUGymWrapper
env_id = "PickCube-v1"
num_eval_envs = 8
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
def cpu_make_env(env_id, env_kwargs = dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # add any other wrappers here
        return env
    return thunk
vector_cls = gym.vector.SyncVectorEnv if num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
eval_envs = vector_cls([cpu_make_env(env_id, env_kwargs) for _ in range(num_eval_envs)])

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # note as there are no partial resets, truncated is True for all environments at the same time
    if truncated.any():
        for final_info in info["final_info"]:
            for k, v in final_info["episode"].items():
                eval_metrics[k].append(v)
for k in eval_metrics.keys():
    print(f"{k}_mean: {np.mean(eval_metrics[k])}")
