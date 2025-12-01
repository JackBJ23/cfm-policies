#!/bin/bash

#SBATCH -N 1                # Number of nodes to allocate (1 physical machine)
#SBATCH -n 1                # Number of tasks (1 Slurm task; for MPI you'd increase this)
#SBATCH -c 4                # Number of CPU cores for the task (4 threads)
#SBATCH -t 02:00:00         # Wall-clock time limit (HH:MM:SS)
#SBATCH -J RUN_NAME         # Job name that appears in the queue
#SBATCH -o out.%j           # File to write STDOUT (%j expands to job ID)
#SBATCH -e err.%j           # File to write STDERR
#SBATCH -p seas_gpu,gpu     # Partitions to run on (searches seas_gpu first, then gpu)
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=16000         # Memory allocation in MB (16 GB)

source ~/.bashrc
conda activate diffusion-policy-ms

name="4_0041"
demos1=570

python -m mani_skill.examples.motionplanning.panda.run -e "PlaceCubeOnPlate-v1" \
    -n ${demos1} -o state --num-procs 10 --traj-name ${name} # --add-obstacle --save-video

python -m mani_skill.trajectory.replay_trajectory --traj_path /n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/demos/PlaceCubeOnPlate-v1/motionplanning/${name}.h5 \
--sim_backend "physx_cpu" \
--obs_mode state \
--target_control_mode pd_ee_delta_pos \
--save_traj \
--num_envs 10