import matplotlib
matplotlib.use("Agg")  # important on a cluster (no display)

import cv2
import matplotlib.pyplot as plt
import os

# for fm_id_8:
video_path = "/n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/examples/baselines/diffusion_policy/runs/eval_fmseeds2/RF-fms100-5_0500-0-bs-512-5-2/1/8.mp4"
output_path = "row5_frames.png"
frame_indices = [2, 60, 80, 100, 150]

# for dp_id_3:
video_path = "/n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/examples/baselines/diffusion_policy/runs/eval_fmseeds2/dp-5_0500-5_0501-0-bs-512-115-2/1/3.mp4"
output_path = "row5_dp_frames.png"
frame_indices = [2, 60, 80, 100, 150]

# for fm_ood:
video_path = "/n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/examples/baselines/diffusion_policy/runs/eval_ood_fmseeds/CFM-fms100-5_0500-40-bs-512-5-2/1/2.mp4"
output_path = "row5_ood_fm_frames.png"
frame_indices = [2, 80, 150, 200, 255]

# for dp_ood:
video_path = "/n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/examples/baselines/diffusion_policy/runs/eval_ood_fmseeds/dp-5_0500-5_0501-40-bs-512-115-2/1/5.mp4"
output_path = "row5_ood_dp_frames.png"
frame_indices = [2, 80, 150, 200, 255]

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

frames = []

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: could not read frame {idx}")
        continue

    # BGR -> RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append((idx, frame_rgb))

cap.release()

n = len(frames)
if n == 0:
    raise RuntimeError("No frames were read from the video.")

# Make a single-row figure
fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

if n == 1:
    axes = [axes]

for ax, (idx, img) in zip(axes, frames):
    ax.imshow(img)
    ax.set_title(f"Frame {idx}")
    ax.axis("off")

plt.tight_layout()

# Save instead of show()
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"Saved montage to {os.path.abspath(output_path)}")

