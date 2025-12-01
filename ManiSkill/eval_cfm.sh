#!/bin/bash

#SBATCH -N 1                # Number of nodes to allocate (1 physical machine)
#SBATCH -n 1                # Number of tasks (1 Slurm task; for MPI you'd increase this)
#SBATCH -c 4                # Number of CPU cores for the task (4 threads)
#SBATCH -t 2:30:00          # Wall-clock time limit (HH:MM:SS)
#SBATCH -J RUN_NAME         # Job name that appears in the queue
#SBATCH -o out.%j           # File to write STDOUT (%j expands to job ID)
#SBATCH -e err.%j           # File to write STDERR
#SBATCH -p seas_gpu,gpu     # Partitions to run on (searches seas_gpu first, then gpu)
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=64000         # Memory allocation in MB (64 GB)

source ~/.bashrc
conda activate dp-cfm1
cd examples/baselines/diffusion_policy

#modelname="CFM-fms100-5_0500-0-bs-512-5"
modelname="dp-5_0500-5_0501-0-bs-512-115"
agent_type="DP"
python evaluations_cfm.py \
    --env-id StackWithObstacle-v1 \
    --checkpoint "runs/${modelname}/checkpoints/best_eval_success_at_end.pt" \
    --modelname ${modelname} \
    --agent_type ${agent_type} \
    --control-mode pd_ee_delta_pos \
    --sim-backend physx_cpu \
    --max-episode-steps 300

# --capture_video \
# --folderout ${modelname} \
