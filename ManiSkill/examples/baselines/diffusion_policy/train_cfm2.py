ALGO_NAME = 'BC_Diffusion_state_UNet'

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusion_policy.evaluate import evaluate

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn
from diffusion_policy.make_env import make_eval_envs
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from dataclasses import dataclass, field
from typing import Optional, List
import tyro

## imports from conditional-flow-matching/examples/images/cifar10/train_cifar10.py:
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    
    ## added: for the second dataset:
    demo_path2: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    num_demos2: Optional[int] = None
    """number of trajectories to load from the second demo dataset"""
    ## until here

    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    agent_type: str = 'DP'
    """agent type: "OTCFM", "RF", "RFpi0", "CFM", "CFMpi0" or "DP" for diffusion policy"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2 # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8 # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64 # not very important
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256]) # default setting is about ~4.5M params
    n_groups: int = 8 # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    fm_steps: int = 30

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

## added function to load and mix two datasets:
def load_and_mix(ds1_path, n1, ds2_path, n2, shuffle=True, seed=0):
    from diffusion_policy.utils import load_demo_dataset
    ds1 = load_demo_dataset(ds1_path, num_traj=n1, concat=False)
    ds2 = load_demo_dataset(ds2_path, num_traj=n2, concat=False)

    # sanity: same keys and types
    assert set(ds1.keys()) == set(ds2.keys()), "dataset keys differ"
    for k in ds1.keys():
        assert isinstance(ds1[k], list) and isinstance(ds2[k], list), f"{k} must be a list"

    # concatenate per-key
    mixed = {k: ds1[k] + ds2[k] for k in ds1.keys()}

    # optional: shuffle trajectories in sync across all fields
    if shuffle:
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(mixed["actions"]))  # one index per trajectory
        for k in mixed.keys():
            mixed[k] = [mixed[k][i] for i in idx]

    return mixed

class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_path, device, num_traj, data_path2, num_traj2):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from diffusion_policy.utils import load_demo_dataset
            ## original: has this line: (changed by load_and_mix below to combine two datasets into one)
            ## trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim),
            ## the list contains num_traj elements, L is trajectory length
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
            
            if num_traj2 is None or num_traj2 == 0:
                trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            else:
                ## added to mix two datasets:
                trajectories = load_and_mix(data_path, num_traj, data_path2, num_traj2, shuffle=True, seed=0)
            ## changed until here
            
        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        # start+self.obs_horizon is at least 1
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0: # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L: # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end-L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)
    
class Agent_OTCFM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        # Velocity field network (same backbone as before)
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        # Time embedding scale (map t in [0, 1] to diffusion-step-like range)
        self.t_embed_scale = 999.0

        # Number of integration steps at test time
        self.fm_steps = args.fm_steps

        # --- TorchCFM OT-CFM loss object ---
        # Standard choice: sigma=0.0 as in the CIFAR example
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    def _embed_t(self, t_b):
        # t_b: (B,) in [0, 1]
        return (t_b * self.t_embed_scale).float()

    def compute_loss(self, obs_seq, action_seq):
        """
        obs_seq:   (B, obs_horizon, obs_dim)
        action_seq:(B, pred_horizon, act_dim)  -> treated as x1 (data)
        """
        B = obs_seq.shape[0]
        device = action_seq.device

        # Conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # Define endpoints for the flow:
        # x0 ~ N(0, I), x1 = action sequence
        x1 = action_seq
        x0 = torch.randn_like(x1)

        # TorchCFM: sample (t, x_t, u_t) along the probability path
        # t:  (B,) or (B,1)   (continuous time in [0, 1])
        # xt: (B, pred_horizon, act_dim)
        # ut: (B, pred_horizon, act_dim)  target vector field
        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)

        # Make sure t is shape (B,)
        t_b = t.view(B, -1)[:, 0]
        t_emb = self._embed_t(t_b)

        # Predict velocity with conditioned UNet
        pred_v = self.velocity_net(xt, t_emb, global_cond=obs_cond)

        # CFM loss: MSE between predicted and target vector field
        return F.mse_loss(pred_v, ut)

    @torch.no_grad()
    def get_action(self, obs_seq, fm_steps=None, use_heun=True):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # 2) Integrate forward: t = 0 -> 1
        t_grid = torch.linspace(0.0, 1.0, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a = a[:, start:end]
        return a # torch.clamp(a, -1.0, 1.0)

class Agent_RF(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.velocity_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        self.t_embed_scale = 999.0
        self.fm_steps = args.fm_steps

        # I-CFM with sigma=0.0 -> 1-Rectified Flow
        self.fm = ConditionalFlowMatcher(sigma=0.0)

    def _embed_t(self, t_b):
        return (t_b * self.t_embed_scale).float()

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]
        device = action_seq.device

        obs_cond = obs_seq.flatten(start_dim=1)

        x1 = action_seq
        x0 = torch.randn_like(x1)

        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)
        t_b = t.view(B, -1)[:, 0]
        t_emb = self._embed_t(t_b)

        pred_v = self.velocity_net(xt, t_emb, global_cond=obs_cond)
        return F.mse_loss(pred_v, ut)

    @torch.no_grad()
    def get_action(self, obs_seq, fm_steps=None, use_heun=True):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # 2) Integrate forward: t = 0 -> 1
        t_grid = torch.linspace(0.0, 1.0, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a = a[:, start:end]
        return a # torch.clamp(a, -1.0, 1.0)

class Agent_RFpi0(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.velocity_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        self.t_embed_scale = 999.0
        self.fm_steps = args.fm_steps

        # I-CFM with sigma=0.0 -> 1-Rectified Flow
        self.fm = ConditionalFlowMatcher(sigma=0.0)

        # ---- NEW: custom timestep sampler parameters (same as CFMpi0) ----
        self.t_max = 0.999  # same s as in Agent_CFMpi0
        self.beta_a = 1.5
        self.beta_b = 1.0

    def _embed_t(self, t_b):
        return (t_b * self.t_embed_scale).float()

    def _sample_timesteps(self, batch_size, device):
        """
        Sample t ~ p(t) induced by:
            u ~ Beta(a=1.5, b=1.0)
            t = t_max * (1 - u)
        so t is concentrated near 0.
        """
        beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        u = beta_dist.sample((batch_size,)).to(device)   # (B,)
        t = self.t_max * (1.0 - u)                       # (B,)
        return t

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]
        device = action_seq.device

        obs_cond = obs_seq.flatten(start_dim=1)

        x1 = action_seq
        x0 = torch.randn_like(x1)

        # ---- NEW: sample times from the same Beta schedule ----
        t_vec = self._sample_timesteps(B, device)   # (B,)
        t_for_fm = t_vec.view(B, 1)                 # (B,1) for TorchCFM

        # Pass t into ConditionalFlowMatcher
        t_out, xt, ut = self.fm.sample_location_and_conditional_flow(
            x0, x1, t=t_for_fm
        )

        # use our sampled t_vec as network time input
        t_b = t_vec
        t_emb = self._embed_t(t_b)

        pred_v = self.velocity_net(xt, t_emb, global_cond=obs_cond)
        return F.mse_loss(pred_v, ut)

    @torch.no_grad()
    def get_action(self, obs_seq, fm_steps=None, use_heun=True):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # 2) Integrate forward: t = 0 -> t_max (same horizon as training)
        t_grid = torch.linspace(0.0, self.t_max, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a = a[:, start:end]
        return a

class Agent_CFM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.velocity_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        self.t_embed_scale = 999.0
        self.fm_steps = args.fm_steps

        # Target CFM (Lipman et al.) from TorchCFM
        self.fm = TargetConditionalFlowMatcher(sigma=0.0)

    def _embed_t(self, t_b):
        return (t_b * self.t_embed_scale).float()

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]
        device = action_seq.device

        obs_cond = obs_seq.flatten(start_dim=1)

        # For Target CFM, x1 is the data; x0 is sampled inside TorchCFM
        # but the API still expects (x0, x1). We pass x0 ~ N(0, I).
        x1 = action_seq
        x0 = torch.randn_like(x1)

        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)
        t_b = t.view(B, -1)[:, 0]
        t_emb = self._embed_t(t_b)

        pred_v = self.velocity_net(xt, t_emb, global_cond=obs_cond)
        return F.mse_loss(pred_v, ut)

    @torch.no_grad()
    def get_action(self, obs_seq, fm_steps=None, use_heun=True):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # 2) Integrate forward: t = 0 -> 1
        t_grid = torch.linspace(0.0, 1.0, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a = a[:, start:end]
        return a # torch.clamp(a, -1.0, 1.0)

class Agent_CFMpi0(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.velocity_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        # Time embedding scaling (keep as in your original code)
        self.t_embed_scale = 999.0

        # Number of solver steps at inference
        self.fm_steps = args.fm_steps

        # Target CFM (Lipman et al.) from TorchCFM
        self.fm = TargetConditionalFlowMatcher(sigma=0.0)

        # ---- NEW: custom timestep sampler parameters ----
        self.t_max = 0.999  # s in your notation
        self.beta_a = 1.5
        self.beta_b = 1.0

    def _embed_t(self, t_b):
        # t_b: (B,) or (B,1)
        return (t_b * self.t_embed_scale).float()

    def _sample_timesteps(self, batch_size, device):
        """
        Sample t ~ p(t) induced by:
            u ~ Beta(a=1.5, b=1.0)
            t = s * (1 - u),  with s = 0.999
        So t is concentrated near 0.
        """
        beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        u = beta_dist.sample((batch_size,)).to(device)          # (B,)
        t = self.t_max * (1.0 - u)                              # (B,)
        return t

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]
        device = action_seq.device

        # Flatten observations for conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # For Target CFM, x1 is the data; x0 is from base N(0, I)
        x1 = action_seq                           # e.g. (B, pred_horizon, act_dim)
        x0 = torch.randn_like(x1)                 # base samples

        # ---- NEW: sample times t from custom distribution ----
        t_vec = self._sample_timesteps(B, device)  # (B,)
        # Shape them as (B, 1, 1) to broadcast over sequence dims if needed
        # depending on what TorchCFM expects. Commonly (B, 1) is enough.
        t_for_fm = t_vec.view(B, 1)               # adjust to (B, 1, 1) if needed

        # ---- Call flow matcher with externally provided t ----
        # we use the fact that TargetConditionalFlowMatcher supports `t` arg.
        t_out, xt, ut = self.fm.sample_location_and_conditional_flow(
            x0, x1, t=t_for_fm
        )

        # We'll trust t_out == t_for_fm; but we already have t_vec, so use that
        t_b = t_vec  # (B,)

        # Time embedding for the network
        t_emb = self._embed_t(t_b)  # (B,)

        # The UNet expects t_emb to broadcast over the action sequence.
        # If your implementation needs (B, 1) or similar, reshape here.
        pred_v = self.velocity_net(xt, t_emb, global_cond=obs_cond)

        return F.mse_loss(pred_v, ut)

    @torch.no_grad()
    def get_action(self, obs_seq, fm_steps=None, use_heun=True):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # 2) Integrate forward: t = 0 -> t_max (or 1.0; here we use t_max for consistency)
        t_grid = torch.linspace(0.0, self.t_max, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            if not use_heun:
                # Euler
                a = a + v1 * dt
            else:
                # Heun's method (2nd-order)
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        # Extract the action horizon segment
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a = a[:, start:end]

        return a  # optionally clamp if needed

class Agent_DP(nn.Module): # standard diffusion policy Agent class
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

def make_agent(env, args) -> nn.Module:
    if args.agent_type == "OTCFM":
        return Agent_OTCFM(env, args)
    elif args.agent_type == "CFM":
        return Agent_CFM(env, args)
    elif args.agent_type == "RF":
        return Agent_RF(env, args)
    elif args.agent_type == "DP":
        return Agent_DP(env, args)
    elif args.agent_type == "CFMpi0":
        return Agent_CFMpi0(env, args)
    elif args.agent_type == "RFpi0":
        return Agent_RFpi0(env, args)
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    ## env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"))
    ## new env_kwargs: self.cubeA_halfsize = float(kwargs.pop("cubeA_halfsize", 0.02))
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", 
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"), 
                cubeA_halfsize=0.02, cubeB_halfsize=0.02, env_x_limit=[-0.2, -0.1], env_y_limit=[-0.15, 0.15],
                offset=[0.0, 0.0], num_obstacles=0, cubeB_offset=[0.15, 0.0])

    ## offset=[0.05, -0.15]: x is positive is further away from robot front, y negative is more to the left
    env_kwargs_ood_pos = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state",
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"),
                cubeA_color=[1, 0, 0, 1], xy_range=0.00, offset=[0.0, -0.15])

    ## to specify
    env_kwargs_ood_color = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state",
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"),
                cubeA_halfsize=0.02, cubeB_halfsize=0.04, cubeA_color=[1, 1, 1, 1], offset=[0.2, 0.0])

    ## to specify
    env_kwargs_ood_sizes = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", 
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"), 
                cubeA_halfsize=0.02, cubeB_halfsize=0.04, cubeA_color=[1,1,1,1], offset=[0.2,0.0])
    
    ## done
    env_kwargs_ood_obstacle = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state",
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"), 
                cubeA_halfsize=0.02, cubeB_halfsize=0.02, env_x_limit=[-0.2, -0.1], env_y_limit=[-0.15, 0.15],
                offset=[0.0, 0.0], num_obstacles=1, cubeB_offset=[0.15, 0.0])
    
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None)
    ## for OOD evaluation on different object positions
    ## the only change wrp to envs above is env_kwargs_ood_pos: this changes the object positions to be out-of-distribution
    env_ood_pos = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs_ood_pos, other_kwargs,  video_dir=f'runs/{run_name}/videos_ood_pos' if args.capture_video else None)
    ## for OOD evaluation on different object colors
    env_ood_color = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs_ood_color, other_kwargs,  video_dir=f'runs/{run_name}/videos_ood_color' if args.capture_video else None)
    ## for OOD evaluation on different object sizes
    env_ood_sizes = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs_ood_sizes, other_kwargs,  video_dir=f'runs/{run_name}/videos_ood_sizes' if args.capture_video else None)
    ## for OOD evaluation on different obstacles
    env_ood_obstacle = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs_ood_obstacle, other_kwargs,  video_dir=f'runs/{run_name}/videos_ood_obstacle' if args.capture_video else None)

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # dataloader setup
    """
    Expanation of the dataset and dataloader:
    - The SmallDemoDataset_DiffusionPolicy loads all demonstration trajectories into GPU memory.
    Each data point is a tuple of (obs_seq, act_seq), where
    obs_seq has shape (obs_horizon, obs_dim)
    act_seq has shape (pred_horizon, act_dim)
    - RandomSampler: Creates an epoch-like pass over the dataset: it will iterate through all indices 
    in random order (no replacement) once.
    - BatchSampler: Groups those indices into mini-batches of size batch_size. 
    - IterationBasedBatchSampler: Wraps the previous sampler but forces it to produce exactly
    total_iters batches, no matter how big or small the dataset is. When it runs out of data,
    it reshuffles and starts over automatically until it reaches total_iters.
    - train_dataloader: loads those batches to the training loop.
    -> so there's no notion of epoch. However, each data point will be used approximately
    total_iters * batch_size / dataset_size times during training.
    """
    ## in the experiments: total_iters * batch_size / dataset_size:
    ## 100000 * 512 / 25068 = 2042.4
    ## 25068 makes sense for total dataset size (ie total obs sequences) since 200*125 = 25k
    ## 200+100 ood obst: 46948. (\simeq 25k + 100*200)
    dataset = SmallDemoDataset_DiffusionPolicy(args.demo_path, device, args.num_demos, args.demo_path2, args.num_demos2)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = len(dataset)

    # agent setup
    agent = make_agent(envs, args).to(device)

    optimizer = optim.AdamW(params=agent.parameters(),
        lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = make_agent(envs, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

    def evaluate_ood_pos(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, env_ood_pos, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"OOD-pos env: Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval_ood_pos/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    ## dont save ckpt for ood eval
                    ## save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"OOD pos env: New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

    def evaluate_ood_obstacle(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, env_ood_obstacle, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"OOD-obstacle env: Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval_ood_obstacle/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    ## dont save ckpt for ood eval
                    ## save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"OOD-obstacle env: New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)
        evaluate_ood_obstacle(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    evaluate_ood_obstacle(args.total_iters)
    log_metrics(args.total_iters)

    envs.close()
    env_ood_obstacle.close()
    writer.close()



