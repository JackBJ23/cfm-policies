import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlaceCubeOnPlateEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: PlaceCubeOnPlateEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    OBSTACLE_HEIGHT = 0.10   # meters
    MOVE_HEIGHT = OBSTACLE_HEIGHT - 0.02 # move around the side of the object
    CLEARANCE = 0.01        # extra margin above the obstacle
    SIDE_STEP = 0.1         # perpendicular detour distance (m)

    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Try a few yaw angles for the grasp
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # NEW: high hover above cube A before any descent (avoid 0.08 m obstacle)
    # -------------------------------------------------------------------------- #
    # Compute cube A top and choose hover_z >= obstacle height + margin
    cube_half_z = float(env.cube_half_size[2].item())
    cubeA_center = env.cubeA.pose.p.cpu().numpy()[0]
    cubeA_top_z = cubeA_center[2] + cube_half_z
    hover_z = max(MOVE_HEIGHT + CLEARANCE, cubeA_top_z + 0.15)
    hover_z_low = cubeA_top_z + 0.15

    # Hover directly above grasp XY at hover_z, same orientation as grasp
    hover_pose = sapien.Pose([grasp_pose.p[0], grasp_pose.p[1], hover_z], grasp_pose.q)
    planner.move_to_pose_with_screw(hover_pose)

    # -------------------------------------------------------------------------- #
    # Reach (descend from hover to just above the grasp pose)
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    # Lift to hover_z again
    lift_pose = sapien.Pose([grasp_pose.p[0], grasp_pose.p[1], hover_z_low], grasp_pose.q)
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Perpendicular detour: side step -> straight -> undo side step
    # -------------------------------------------------------------------------- #
    A_p = env.cubeA.pose.p.cpu().numpy()[0]  # (3,)
    B_p = env.cubeB.pose.p.cpu().numpy()[0]  # (3,)

    # Direction A->B in XY
    d = B_p[:2] - A_p[:2]
    n = np.linalg.norm(d)
    dir_xy = np.array([1.0, 0.0]) if n < 1e-8 else d / n

    # Perpendicular (left/right). Deterministic if seed provided.
    rng = np.random.default_rng(seed)
    side = float(rng.choice([-1.0, 1.0]))
    perp_xy = side * np.array([-dir_xy[1], dir_xy[0]])

    lateral_xy = SIDE_STEP * perp_xy

    # Original align logic (target over B at stack height)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()])
    offset = (goal_pose.p - env.cubeA.pose.p).cpu().numpy()[0]  # (3,)

    # 1) Side step at hover height
    via_side1 = sapien.Pose(
        [lift_pose.p[0] + lateral_xy[0], lift_pose.p[1] + lateral_xy[1], hover_z_low],
        lift_pose.q,
    )
    planner.move_to_pose_with_screw(via_side1)

    # 2) Straight toward B at hover height (keep lateral offset)
    via_straight = sapien.Pose(
        [via_side1.p[0] + offset[0], via_side1.p[1] + offset[1], hover_z_low],
        lift_pose.q,
    )
    planner.move_to_pose_with_screw(via_straight)

    # 3) Undo lateral offset to align exactly above B
    align_pose = sapien.Pose(
        [via_straight.p[0] - lateral_xy[0], via_straight.p[1] - lateral_xy[1], hover_z_low],
        lift_pose.q,
    )
    planner.move_to_pose_with_screw(align_pose)

    # -------------------------------------------------------------------------- #
    # Descend to place and open
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(goal_pose.p.cpu().numpy()[0], align_pose.q)
    planner.move_to_pose_with_screw(place_pose)

    res = planner.open_gripper()
    planner.close()
    return res

'''
import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlaceCubeOnPlateEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: PlaceCubeOnPlateEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Detour perpendicular to A->B (±0.2 m), then straight, then undo lateral offset
    # -------------------------------------------------------------------------- #
    # Positions are batched tensors; take [0] and convert to numpy
    A_p = env.cubeA.pose.p.cpu().numpy()[0]  # (3,)
    B_p = env.cubeB.pose.p.cpu().numpy()[0]  # (3,)

    # Direction from A to B in XY
    d = B_p[:2] - A_p[:2]
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        # Degenerate case: pick an arbitrary axis
        dir_xy = np.array([1.0, 0.0], dtype=float)
    else:
        dir_xy = d / norm

    # Perpendicular direction (left/right)
    # Left of dir_xy is [-y, x]; right is [y, -x]. We choose ± randomly.
    rng = np.random.default_rng(seed)
    side = float(rng.choice([-1.0, 1.0]))
    perp_xy = side * np.array([-dir_xy[1], dir_xy[0]], dtype=float)

    lateral_mag = 0.1  # meters
    lateral_xy = lateral_mag * perp_xy

    # Compute the XY translation from lift to align (toward B)
    # We'll define align_pose below, but the XY offset is simply (B - A) projected
    # from the original code's offset:
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()])
    offset = (goal_pose.p - env.cubeA.pose.p).cpu().numpy()[0]  # (3,)

    # Pose directly above A (after lift)
    lift_xyz = lift_pose.p
    lift_q = lift_pose.q

    # 1) Side step at current height
    via_side1_xyz = np.array([lift_xyz[0] + lateral_xy[0],
                              lift_xyz[1] + lateral_xy[1],
                              lift_xyz[2]], dtype=float)
    via_side1 = sapien.Pose(via_side1_xyz, lift_q)

    # 2) Move straight toward B while maintaining lateral offset
    # add the XY part of 'offset' (from A to above-B) at same height
    via_straight_xyz = np.array([via_side1_xyz[0] + offset[0],
                                 via_side1_xyz[1] + offset[1],
                                 lift_xyz[2]], dtype=float)
    via_straight = sapien.Pose(via_straight_xyz, lift_q)

    # 3) Undo the lateral offset to align exactly above B (same height as lift)
    align_xyz = np.array([via_straight_xyz[0] - lateral_xy[0],
                          via_straight_xyz[1] - lateral_xy[1],
                          lift_xyz[2]], dtype=float)
    align_pose = sapien.Pose(align_xyz, lift_q)

    # Execute the three legs
    planner.move_to_pose_with_screw(via_side1)
    planner.move_to_pose_with_screw(via_straight)
    planner.move_to_pose_with_screw(align_pose)

    # -------------------------------------------------------------------------- #
    # Descend and deposit at the stack height over B
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(goal_pose.p.cpu().numpy()[0], align_pose.q)
    planner.move_to_pose_with_screw(place_pose)

    res = planner.open_gripper()
    planner.close()
    return res
'''