from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common

def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n: ## n = args.num_eval_episodes (default 100)
            ## Means we want to collect 100 episodes of evaluation data
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics

## added function (takes in another metric: straightness of path)
def evaluate_straightness(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)

    straightness_list = []  # collect straightness over all eval episodes

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:  # n = args.num_eval_episodes (default 100)
            # Means we want to collect 100 episodes of evaluation data
            obs = common.to_tensor(obs, device)

            # agent.get_action now returns (action_seq, straightness)
            action_seq, straightness = agent.get_action(obs)  # straightness: (B,)

            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), \
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"

                # log environment episode metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                # log straightness for these episodes (one per env)
                # straightness is a tensor of shape (num_envs,)
                straightness_list.append(straightness.detach().cpu().numpy())

                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

    agent.train()

    # stack env metrics
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])

    # aggregate straightness into a single scalar
    if len(straightness_list) > 0:
        straightness_all = np.concatenate(straightness_list, axis=0)  # shape (num_episodes * num_envs,)
        straightness_mean = float(straightness_all.mean())
    else:
        straightness_mean = 0.0

    return eval_metrics, straightness_mean

## added function (takes in another metric: straightness of path)
def evaluate_energy(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)

    straightness_list = []  # collect straightness over all eval episodes
    energy_list = []

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:  # n = args.num_eval_episodes (default 100)
            # Means we want to collect 100 episodes of evaluation data
            obs = common.to_tensor(obs, device)

            # agent.get_action now returns (action_seq, straightness)
            action_seq, straightness, energy = agent.get_action(obs)  # straightness: (B,)

            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), \
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"

                # log environment episode metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                # log straightness for these episodes (one per env)
                # straightness is a tensor of shape (num_envs,)
                straightness_list.append(straightness.detach().cpu().numpy())
                energy_list.append(energy.detach().cpu().numpy())

                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

    agent.train()

    # stack env metrics
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])

    # aggregate straightness into a single scalar
    if len(straightness_list) > 0:
        straightness_all = np.concatenate(straightness_list, axis=0)  # shape (num_episodes * num_envs,)
        straightness_mean = float(straightness_all.mean())
    else:
        straightness_mean = 0.0

    # aggregate energy into a single scalar
    if len(energy_list) > 0:
        energy_all = np.concatenate(energy_list, axis=0)  # shape (num_episodes * num_envs,)
        energy_mean = float(energy_all.mean())
    else:
        energy_mean = 0.0

    return eval_metrics, straightness_mean, energy_mean
