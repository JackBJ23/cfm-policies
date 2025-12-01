#!/bin/bash

#SBATCH -N 1                # Number of nodes to allocate (1 physical machine)
#SBATCH -n 1                # Number of tasks (1 Slurm task; for MPI you'd increase this)
#SBATCH -c 4                # Number of CPU cores for the task (4 threads)
#SBATCH -t 00:10:00         # Wall-clock time limit (HH:MM:SS)
#SBATCH -J RUN_NAME         # Job name that appears in the queue
#SBATCH -o out.%j           # File to write STDOUT (%j expands to job ID)
#SBATCH -e err.%j           # File to write STDERR
#SBATCH -p seas_gpu,gpu     # Partitions to run on (searches seas_gpu first, then gpu)
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=16000         # Memory allocation in MB (16 GB)

conda activate diffusion-policy-ms
python -m mani_skill.examples.motionplanning.panda.run -e "PlaceCubeOnPlate-v1" -n 10 --num-procs 10 \
    -o state --traj-name 1_0002 --save-video \





