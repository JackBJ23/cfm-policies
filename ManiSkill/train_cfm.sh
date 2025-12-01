#!/bin/bash

#SBATCH -N 1                # Number of nodes to allocate (1 physical machine)
#SBATCH -n 1                # Number of tasks (1 Slurm task; for MPI you'd increase this)
#SBATCH -c 4                # Number of CPU cores for the task (4 threads)
#SBATCH -t 15:00:00         # Wall-clock time limit (HH:MM:SS)
#SBATCH -J RUN_NAME         # Job name that appears in the queue
#SBATCH -o out.%j           # File to write STDOUT (%j expands to job ID)
#SBATCH -e err.%j           # File to write STDERR
#SBATCH -p seas_gpu,gpu     # Partitions to run on (searches seas_gpu first, then gpu)
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=64000         # Memory allocation in MB (64 GB)

source ~/.bashrc
conda activate dp-cfm1 # diffusion-policy-ms
cd examples/baselines/diffusion_policy

fm_steps=100 # 30 default
agent_type="CFM" # "OTCFM", "RF", "RFpi0", "CFM", "CFMpi0", or "DP"
train_seed=115
batch_size=512
# for OOD obstacle:
ds_name1="5_0500" # without obstacle
ds_name2="5_0501" # with obstacle
demos1=200

# note: if using OOD-pose: use train.py and set max_episode_steps 150
# if using OOD-obstacle, use train3.py and set max_episode_steps 300
for demos2 in 20; do
  echo "=== Running demos2=${demos2} ==="
  srun python train_cfm2.py \
    --env-id StackWithObstacle-v1 \
    --demo-path  /n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/demos_${ds_name1}/StackWithObstacle-v1/motionplanning/${ds_name1}.state.pd_ee_delta_pos.physx_cpu.h5 \
    --demo-path2 /n/home08/jackbjed/scaling-diffusion-policy/ManiSkill/demos_${ds_name2}/StackWithObstacle-v1/motionplanning/${ds_name2}.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode pd_ee_delta_pos \
    --sim-backend physx_cpu \
    --num-demos ${demos1} \
    --num-demos2 ${demos2} \
    --total_iters 100000 \
    --batch_size ${batch_size} \
    --exp-name ${agent_type}-fms${fm_steps}-${ds_name1}-${demos2}-bs-${batch_size}-${train_seed} \
    --eval_freq 5000 \
    --max_episode_steps 300 \
    --agent_type ${agent_type} \
    --cuda \
    --track \
    --fm_steps ${fm_steps} \
    --seed ${train_seed}
done

