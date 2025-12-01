# cfm-policies

This repository contains code, scripts, and experiments for training and evaluating **CFM-based policies** as well as **Diffusion Policy**, built on top of the ManiSkill simulation framework. It includes utilities for dataset creation, training, evaluation, and plotting, along with modified versions of ManiSkill baseline scripts. In particular, this repository introduces a new environment and motion-planning setup for stacking a red cube on top of a green cube in the presence of an obstacle (a blue block) positioned between them.

## Contents
- `ManiSkill/` — main source code and experiment scripts  
- Training + evaluation scripts (CFM, RF, DP, and variants)  
- Plotting utilities for straightness, energy, and other metrics  
- Helper scripts for dataset generation and running experiments

## Usage
Clone the repository:
```bash
git clone https://github.com/JackBJ23/cfm-policies.git
cd cfm-policies
```
Next, create the conda environment `diffusion-policy-ms` following the instructions [here](https://github.com/JackBJ23/cfm-policies/tree/main/ManiSkill/examples/baselines/diffusion_policy) (replace the environment name `diffusion-policy-ms` by `dp-cfm1`), and after installing the environment, install torchcfm with
```bash
pip install torchcfm
```

## Note

Due to our configuration of the repository, some .sh files show paths to folders (e.g., datasets, checkpoints)
that should be changed based on the true full paths of these documents.

 ## 
