#!/bin/bash

#SBATCH -N 1                # Number of nodes to allocate (1 physical machine)
#SBATCH -n 1                # Number of tasks (1 Slurm task; for MPI you'd increase this)
#SBATCH -c 4                # Number of CPU cores for the task (4 threads)
#SBATCH -t 05:00:00         # Wall-clock time limit (HH:MM:SS)
#SBATCH -J RUN_NAME         # Job name that appears in the queue
#SBATCH -o out.%j           # File to write STDOUT (%j expands to job ID)
#SBATCH -e err.%j           # File to write STDERR
#SBATCH -p seas_gpu,gpu     # Partitions to run on (searches seas_gpu first, then gpu)
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=64000         # Memory allocation in MB (64 GB)

source ~/.bashrc
conda activate diffusion-policy-ms
cd examples/baselines/diffusion_policy

train_seed=11
ds_name1="1_0002"
ds_name2="1_0004"
demos1=200
demos2=0
python train.py --env-id PlaceCubeOnPlate-v1 \
  --demo-path /n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/demos/PlaceCubeOnPlate-v1/motionplanning/${ds_name1}.state.pd_ee_delta_pos.physx_cpu.h5 \
  --demo-path2 /n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/demos/PlaceCubeOnPlate-v1/motionplanning/${ds_name2}.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" \
  --num-demos ${demos1} --num-demos2 ${demos2} \
  --total_iters 40000 \
  --batch_size 256 \
  --exp-name dp-${ds_name1}-${ds_name2}-${demos2}-${train_seed} \
  --eval_freq 5000 \
  --max_episode_steps 150 \
  --cuda \
  --track