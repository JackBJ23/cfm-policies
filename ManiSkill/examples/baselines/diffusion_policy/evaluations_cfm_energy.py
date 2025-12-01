ALGO_NAME = 'BC_Diffusion_state_UNet'

import os
import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List
import tyro

from diffusion_policy.evaluate import evaluate, evaluate_straightness, evaluate_energy
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

# ---------- Args (eval-only) ----------
@dataclass
class Args:
    # required
    checkpoint: str
    """Path to the model checkpoint file to load (e.g., runs/.../checkpoints/best.pt)."""
    modelname: str
    agent_type: str

    # core eval knobs
    env_id: str = "StackWithObstacle-v1"
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_ee_delta_pos" # "pd_joint_delta_pos"
    max_episode_steps: int = 300
    capture_video: bool = False  # set True to save eval videos
    folderout: str = "default" # folder name suffix for video saving

    # diffusion policy architecture (must match training)
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    # device / seed
    cuda: bool = True
    seed: int = 1
    torch_deterministic: bool = True

# Agents

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
        self.fm_steps = 30

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
    def get_action(self, obs_seq, fm_steps=None, use_heun=False):
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
        self.fm_steps = 30

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
    def get_action(self, obs_seq, fm_steps=None, use_heun=False):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        a_initial = a.clone()  # for straight-line distance

        # track total path length per batch element
        path_length = torch.zeros(B, device=device)

        # 2) Integrate forward: t = 0 -> 1
        t_grid = torch.linspace(0.0, 1.0, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            # contribution to path length from this step: ||v1 * dt||
            step_disp = v1 * dt                       # (B, pred_horizon, act_dim)
            path_length += step_disp.reshape(B, -1).norm(dim=1)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        # full final trajectory (before slicing)
        a_final = a

        # slice to action horizon for control
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a_out = a[:, start:end]

        # straight-line distance between initial and final states
        straight_dist = (a_final - a_initial).reshape(B, -1).norm(dim=1)  # (B,)

        # avoid division by zero
        eps = 1e-8
        straightness = straight_dist / (path_length + eps)                # (B,)

        # return both action sequence and straightness metric
        return a_out, straightness

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
        self.fm_steps = 30

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
    def get_action(self, obs_seq, fm_steps=None, use_heun=False):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        a_initial = a.clone()  # save for straight-line distance

        # track total path length per batch element
        path_length = torch.zeros(B, device=device)

        # 2) Integrate forward: t = 0 -> t_max (same horizon as training)
        t_grid = torch.linspace(0.0, self.t_max, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            # contribution to path length from this step: ||v1 * dt||
            step_disp = v1 * dt                           # (B, pred_horizon, act_dim)
            path_length += step_disp.view(B, -1).norm(dim=1)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        # full final trajectory (before slicing) for straight-line distance
        a_final = a

        # slice to action horizon for control
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a_out = a[:, start:end]

        # straight-line distance between initial and final states
        straight_dist = (a_final - a_initial).view(B, -1).norm(dim=1)  # (B,)

        eps = 1e-8
        straightness = path_length / (straight_dist + eps)             # (B,)

        # return both action and straightness metric
        return a_out, straightness

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
        self.fm_steps = 30

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
    def get_action(self, obs_seq, fm_steps=None, use_heun=False):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        a_initial = a.clone()  # for straight-line distance

        # track total path length per batch element
        path_length = torch.zeros(B, device=device)

        # 2) Integrate forward: t = 0 -> 1
        t_grid = torch.linspace(0.0, 1.0, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            # accumulate path length: ||v1 * dt|| over all dims
            step_disp = v1 * dt                             # (B, pred_horizon, act_dim)
            path_length += step_disp.reshape(B, -1).norm(dim=1)

            if not use_heun:
                a = a + v1 * dt
            else:
                a_euler = a + v1 * dt
                t_b_next = t_next.expand(B)
                t_emb_next = self._embed_t(t_b_next)
                v2 = self.velocity_net(a_euler, t_emb_next, global_cond=obs_cond)
                a = a + 0.5 * (v1 + v2) * dt

        # full final trajectory for straight-line distance
        a_final = a

        # slice to action horizon for control
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a_out = a[:, start:end]

        # straight-line distance between initial and final
        straight_dist = (a_final - a_initial).reshape(B, -1).norm(dim=1)  # (B,)

        eps = 1e-8
        straightness = straight_dist / (path_length + eps)                # (B,)

        return a_out, straightness

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
        self.fm_steps = 30

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
    def get_action(self, obs_seq, fm_steps=None, use_heun=False):
        device = obs_seq.device
        B = obs_seq.shape[0]
        fm_steps = fm_steps or self.fm_steps

        obs_cond = obs_seq.flatten(start_dim=1)

        # 1) Start from noise at t = 0
        a = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        a_initial = a.clone()  # for straight-line distance

        # track total path length per batch element
        path_length = torch.zeros(B, device=device)

        # 2) Integrate forward: t = 0 -> t_max (or 1.0; here we use t_max for consistency)
        t_grid = torch.linspace(0.0, self.t_max, fm_steps + 1, device=device)

        for i in range(fm_steps):
            t = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t   # > 0

            t_b = t.expand(B)
            t_emb = self._embed_t(t_b)
            v1 = self.velocity_net(a, t_emb, global_cond=obs_cond)

            # accumulate path length: ||v1 * dt|| over all dims
            step_disp = v1 * dt                           # (B, pred_horizon, act_dim)
            path_length += step_disp.view(B, -1).norm(dim=1)

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

        # full final trajectory for straight-line distance
        a_final = a

        # Extract the action horizon segment
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        a_out = a[:, start:end]

        # straight-line distance between initial and final
        straight_dist = (a_final - a_initial).view(B, -1).norm(dim=1)  # (B,)

        eps = 1e-8
        straightness = path_length / (straight_dist + eps)             # (B,)

        return a_out, straightness  # optionally clamp a_out if needed

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
        self.num_diffusion_iters = 100 # 100
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
        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        device = obs_seq.device

        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=device
            )

            # save initial for straight-line distance
            x0 = noisy_action_seq.clone()

            # track path length per batch
            path_length = torch.zeros(B, device=device)
            for k in self.noise_scheduler.timesteps:
                x_prev = noisy_action_seq

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

                # accumulate path length: ||x_{k+1} - x_k|| over all dims
                step_disp = (noisy_action_seq - x_prev)          # (B, pred_horizon, act_dim)
                step_disp_flat = step_disp.reshape(B, -1)        # <- reshape instead of view
                path_length += step_disp_flat.norm(dim=1)

            # final sample (before slicing)
            xT = noisy_action_seq

        # straight-line distance between initial and final
        straight_dist = (xT - x0).reshape(B, -1).norm(dim=1)  # (B,)
        eps = 1e-8
        straightness = straight_dist / (path_length + eps)    # (B,)

        # only take act_horizon number of actions for control
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        actions = noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

        return actions, straightness

# ---------- Util: load checkpoint robustly ----------
def load_checkpoint_into_agent(agent: nn.Module, ckpt_path: str, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # common patterns
    candidates = []
    if isinstance(ckpt, dict):
        # try a few likely keys
        for k in ["state_dict", "model", "agent", "ema_state_dict", "ema_model", "ema_agent"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                candidates.append(ckpt[k])
        # whole-dict might already be a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            candidates.insert(0, ckpt)
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    last_err = None
    for sd in candidates:
        try:
            agent.load_state_dict(sd, strict=False)
            print(f"[info] Loaded weights from key set (strict=False) successfully.")
            return
        except Exception as e:
            last_err = e
    # final attempt: strip a prefix like 'module.' or 'noise_pred_net.'
    if candidates:
        sd = candidates[0]
        stripped = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("agent."):
                nk = nk[len("agent."):]
            stripped[nk] = v
        agent.load_state_dict(stripped, strict=False)
        print("[info] Loaded weights after key-prefix stripping (strict=False).")
        return

    raise RuntimeError(f"Could not load checkpoint from {ckpt_path}: {last_err}")

# define evaluation and logging functions
def evaluate_ckpt(agent, args, device, modelname, fm_steps):
    # create envs
    if args.capture_video:
        video_dir = f"runs/eval_fmseeds3/{args.folderout}-2/{fm_steps}"
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", 
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"), 
                cubeA_halfsize=0.02, cubeB_halfsize=0.02, env_x_limit=[-0.2, -0.1], env_y_limit=[-0.15, 0.15],
                offset=[0.0, 0.0], num_obstacles=0, cubeB_offset=[0.15, 0.0])

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    env_pos = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs, other_kwargs, video_dir=video_dir
    )
    if args.agent_type=="DP":
        agent.noise_scheduler.set_timesteps(fm_steps)
        agent.num_diffusion_iters = fm_steps
        '''
        agent.num_diffusion_iters = fm_steps
        agent.noise_scheduler = DDPMScheduler(
            num_train_timesteps=fm_steps,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )
        '''
    else:
        agent.fm_steps = fm_steps
    # Evaluate (ID-pos only)
    t0 = time.time()
    eval_metrics, straightness_mean = evaluate_straightness(
        args.num_eval_episodes, agent, env_pos, device, args.sim_backend
    )
    elapsed = time.time() - t0
    env_pos.close()
    # Aggregate and print
    print(f"\n==={modelname}, fm_steps={fm_steps}===")
    print(f"Evaluated {len(eval_metrics.get('success_at_end', []))} episodes in {elapsed:.1f}s")
    # print per-metric mean
    for k, arr in eval_metrics.items():
        try:
            val = float(np.mean(arr))
        except Exception:
            val = arr
        print(f"{k}: {val}")
    print("Straightness:", straightness_mean)
    return 0

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

# ---------- Main (eval only, incl. OOD-pos env) ----------
if __name__ == "__main__":
    args = tyro.cli(Args)

    # seeding / determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    if args.capture_video:
        video_dir = f"runs/eval_id_pos/videos_{args.folderout}"
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None

    # Base env kwargs (matches the training/eval settings)
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", 
                render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"), 
                cubeA_halfsize=0.02, cubeB_halfsize=0.02, env_x_limit=[-0.2, -0.1], env_y_limit=[-0.15, 0.15],
                offset=[0.0, 0.0], num_obstacles=0, cubeB_offset=[0.15, 0.0])

    other_kwargs = dict(obs_horizon=args.obs_horizon)

    # create environments
    envs = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs, other_kwargs, video_dir=video_dir
    )
    envs.close()
    # agent setup
    agent = make_agent(envs, args).to(device)
    # Load weights
    print(f"[info] Loading checkpoint from: {args.checkpoint}")
    load_checkpoint_into_agent(agent, args.checkpoint, map_location=device)
    agent.eval()

    # evaluate and print metrics
    for fm_steps in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
        # evaluate and print metrics
        evaluate_ckpt(agent, args, device, args.modelname, fm_steps)


