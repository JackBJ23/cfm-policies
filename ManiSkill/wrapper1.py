import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
env = gym.make("PickCube-v1", num_envs=1, render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=30)
env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
