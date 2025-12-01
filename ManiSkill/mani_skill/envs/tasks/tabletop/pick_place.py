from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("PlaceCubeOnPlate-v1", max_episode_steps=150)
class PlaceCubeOnPlateEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cubeA_halfsize = float(kwargs.pop("cubeA_halfsize", 0.02))
        self.cubeB_halfsize = float(kwargs.pop("cubeB_halfsize", 0.02))
        self.cubeA_color = kwargs.pop("cubeA_color", [1, 0, 0, 1])  # red
        self.cubeB_color = kwargs.pop("cubeB_color", [0, 1, 0, 1])  # green
        self.env_x_limit = kwargs.pop("env_x_limit", [-0.15, 0.15]) # [-0.15, -0.05]
        self.env_y_limit = kwargs.pop("env_y_limit", [-0.1, 0.1])
        ## env_y_limit controls the y range where cubes can be spawned, which is the longer side of the table
        ## ie the x axis goes from the robot arm to the opposite side of the table, y axis is left-right
        ## for the random jitter around an anchor point:
        self.extra_margin = kwargs.pop("extra_margin", 0.0) # make it 0.05 if add obstacle
        self.xy_range = kwargs.pop("xy_range", 0.0)
        self.offset = kwargs.pop("offset", [0.0, 0.0])
        ## for obstacles:
        self.num_obstacles = kwargs.pop("num_obstacles", 0)  # set to >0 to add an obstacle
        self.block_halfsize = kwargs.pop("block_halfsize", [0.02, 0.02, 0.05])
        self.block_color = kwargs.pop("block_color", [0.9, 0.6, 0.2, 1])
        ## added for optional movement of objects B to the side:
        self.cubeB_offset = kwargs.pop("cubeB_offset", [0.0, 0.0]) # 0.12
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        '''
        # added:
        # self.cubeA_halfsize = 0.02
        self.cubeB_halfsize = 0.02
        
        self.cubeA_color = [1, 0, 0, 1]  # red
        self.cubeB_color = [0, 1, 0, 1]  # green
        self.env_x_limit = [-0.2, 0.2] # [-0.1, 0.1]
        ## env_y_limit controls the y range where cubes can be spawned, which is the longer side of the table
        ## ie the x axis goes from the robot arm to the opposite side of the table, y axis is left-right
        self.env_y_limit = [-0.2, 0.2] # [-0.15, 0.15]
        ## for the random jitter around an anchor point:
        self.xy_range = 0.0
        self.offset = [0.0, 0.0]
        
        ## for obstacles:
        self.block_halfsize = 0.02
        self.block_color = [0.5, 0.5, 0.5, 1]  # gray
        self.num_obstacles = 0  # set to >0 to add an obstacle cube
        if self.num_obstacles > 0:
            self.env_x_limit = [-0.2, 0.2] # original: [-0.1, 0.1]
            self.env_y_limit = [-0.25, 0.25] # original: [-0.2, 0.2]
        '''
        
        self.cube_half_size = common.to_tensor([self.cubeB_halfsize] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=self.cubeA_halfsize,
            color=self.cubeA_color,
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=self.cubeB_halfsize,
            color=self.cubeB_color,
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        if self.num_obstacles > 0:
            self.block = actors.build_box(
                self.scene,
                half_sizes=self.block_halfsize,    # 20cm × 10cm × 6cm
                color= self.block_color,
                name="block",
                initial_pose=sapien.Pose(p=[1.0, 0.0, 0.1]),
            )
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # Half-sizes (ensure they're torch scalars on the right device)
            hsA = torch.as_tensor(self.cubeA_halfsize, device=self.device, dtype=torch.float32)
            hsB = torch.as_tensor(self.cubeB_halfsize, device=self.device, dtype=torch.float32)

            # Per-object clearance radii (half-diagonal in XY) + small margin
            margin = 1e-3 + self.extra_margin
            radiusA = torch.linalg.norm(torch.stack([hsA, hsA])) + margin
            radiusB = torch.linalg.norm(torch.stack([hsB, hsB])) + margin

            # Base XY jitter around a local anchor
            ## original: * 0.2 - 0.1
            # xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1 # ie random in [-0.1, 0.1] x [-0.1, 0.1] for each env in batch
            ## anchor picks a global location (e.g., anywhere on the table). The local jitter scatters objects near that anchor.
            ## This gives clustered scenes instead of uniformly spraying objects everywhere.
            
            ## reaches 73% success rate: * 0.15 - 0.075
            xy = torch.rand((b, 2), device=self.device) * self.xy_range -  self.xy_range / 2
            ## ie xy is random in [self.self.xy_range/2, self.self.xy_range/2]**2
            offset = torch.tensor(self.offset, device=self.device)
            xy = xy + offset  # shift the center to the offset location

            ## sampling region is [[low x, low y], [up x, up y]]
            # region = [[-0.1, -0.2], [0.1, 0.2]] ## so the region to sample pts is [-0.1, 0.1] x [-0.2, 0.2]
            ## changed to:
            region = [[self.env_x_limit[0], self.env_y_limit[0]], [self.env_x_limit[1], self.env_y_limit[1]]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            # Sample placements with per-cube radii
            ## sampler.sample: Args: radius (float): collision radius, max_trials (int): maximal trials to sample.
            ## adding xy makes the env limits: cube center x in [-0.2, 0.2], y in [-0.3, 0.3]
            ## the sampler ensures no collisions between the two cubes
            cubeA_xy = xy + sampler.sample(radiusA.item(), 100)
            cubeB_offset = torch.tensor(self.cubeB_offset, device=self.device)
            cubeB_xy = xy + sampler.sample(radiusB.item(), 100, verbose=False)
            ## added for optionally translating only cube B:
            cubeB_xy = cubeB_xy + cubeB_offset
            # pose for block C (obstacle):
            if self.num_obstacles > 0:
                hsC = torch.as_tensor(self.block_halfsize[0], device=self.device, dtype=torch.float32)
                xyzC = torch.zeros((b, 3), device=self.device)
                xy = torch.rand((b, 2), device=self.device) * 0.02 -  0.01
                xyzC[:, :2] = (cubeA_xy + cubeB_xy) / 2.0 + xy
                xyzC[:, 2]  = hsC
                qsC = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
                self.block.set_pose(Pose.create_from_pq(p=xyzC, q=qsC))

            # Build per-cube xyz (use each cube's own Z = half-size)
            xyzA = torch.zeros((b, 3), device=self.device)
            xyzA[:, :2] = cubeA_xy
            xyzA[:, 2]  = hsA

            xyzB = torch.zeros((b, 3), device=self.device)
            xyzB[:, :2] = cubeB_xy
            xyzB[:, 2]  = hsB

            # Random yaw (lock x,y)
            qsA = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyzA, q=qsA))

            qsB = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyzB, q=qsB))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
