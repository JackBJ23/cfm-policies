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

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ---------- Args (eval-only) ----------
@dataclass
class Args:
    # required
    checkpoint: str
    """Path to the model checkpoint file to load (e.g., runs/.../checkpoints/best.pt)."""

    # core eval knobs
    env_id: str = "PlaceCubeOnPlate-v1"
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_ee_delta_pos" # "pd_joint_delta_pos"
    max_episode_steps: int = 150
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
    seed: int = 115
    torch_deterministic: bool = True

# ---------- Agent (unchanged, inference-capable) ----------
class Agent(nn.Module):
    def __init__(self, env, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1       # (act_dim,)
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),  # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def compute_loss(self, obs_seq, action_seq):
        # Not used in eval-only
        raise NotImplementedError

    def get_action(self, obs_seq):
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
            # Use scheduler timesteps as set at construction (DDPM with same steps as training)
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=noisy_action_seq
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

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

# ---------- Main (eval only, incl. OOD-pos env) ----------
if __name__ == "__main__":
    args = tyro.cli(Args)

    # seeding / determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    # Base env kwargs (matches the training/eval settings)
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        cubeA_halfsize=0.02, cubeB_halfsize=0.02, offset=[0.0, 0.15],
        env_x_limit=[-0.15, 0.15], env_y_limit=[-0.1, 0.1], cubeB_offset=[0., 0.], extra_margin=0.,
        max_episode_steps=args.max_episode_steps,
    )
    # OOD-pos variant (object positions shifted) (shifted along Y axis to the left)
    env_kwargs_ood_pos1 = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        cubeA_color=[1, 0, 0, 1], xy_range=0.00, offset=[0.0, -0.15],
        env_x_limit=[-0.15, 0.15], env_y_limit=[-0.1, 0.1], cubeB_offset=[0., 0.], extra_margin=0.,
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    # Build one env just to infer shapes for the Agent
    # (We’ll evaluate on OOD-pos envs, but shapes are the same.)
    envs_for_shapes = make_eval_envs(
        args.env_id, 1, args.sim_backend, env_kwargs, other_kwargs,
        video_dir=None
    )
    agent = Agent(envs_for_shapes, args).to(device)
    envs_for_shapes.close()

    # Load weights
    print(f"[info] Loading checkpoint from: {args.checkpoint}")
    load_checkpoint_into_agent(agent, args.checkpoint, map_location=device)
    agent.eval()

    # Build OOD-pos eval envs
    video_dir = None
    video_dir_ood1 = None
    video_dir_ood2 = None
    video_dir_ood3 = None
    if args.capture_video:
        # Keep it simple; write under runs/eval_ood_pos/
        video_dir = f"runs/eval_id_pos/videos_{args.folderout}"
        video_dir_ood1 = f"runs/eval_ood_pos1/videos_{args.folderout}"
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(video_dir_ood1, exist_ok=True)

    ## For each test environment, 1) create environment, 2) evaluate it, 3) print results
    # standard in-dist environment
    env_pos = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs, other_kwargs, video_dir=video_dir
    )
    # Evaluate (ID-pos only)
    t0 = time.time()
    eval_metrics = evaluate(
        args.num_eval_episodes, agent, env_pos, device, args.sim_backend
    )
    elapsed = time.time() - t0
    env_pos.close()
    # Aggregate and print
    print(f"\n=== ID Position Evaluation ===")
    print(f"Evaluated {len(eval_metrics.get('success_at_end', []))} episodes in {elapsed:.1f}s")
    # print per-metric mean
    for k, arr in eval_metrics.items():
        try:
            val = float(np.mean(arr))
        except Exception:
            val = arr
        print(f"{k}: {val}")

    # OOD position environments
    env_ood_pos1 = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs_ood_pos1, other_kwargs, video_dir=video_dir_ood1
    )
    # Evaluate (OOD-pos 1)
    t0 = time.time()
    eval_metrics_ood1 = evaluate(
        args.num_eval_episodes, agent, env_ood_pos1, device, args.sim_backend
    )
    elapsed_ood1 = time.time() - t0
    env_ood_pos1.close()
    print(f"\n=== OOD Position Evaluation 1 ===")
    print(f"Evaluated {len(eval_metrics_ood1.get('success_at_end', []))} episodes in {elapsed_ood1:.1f}s")
    # print per-metric mean
    for k, arr in eval_metrics_ood1.items():
        try:
            val = float(np.mean(arr))
        except Exception:
            val = arr
        print(f"{k}: {val}")

