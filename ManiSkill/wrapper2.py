import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
N = 4
env = gym.make("PickCube-v1", num_envs=N, render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", max_steps_per_video=50, video_fps=30)
env = ManiSkillVectorEnv(env, auto_reset=True) # adds auto reset
env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("done")
