#!/bin/bash

#SBATCH --qos=gpushort
##SBATCH --qos=gpulong
##SBATCH --qos=gpumedium
##SBATCH --qos=gpupreempt

#SBATCH --job-name=opt_dly
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=500G

#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err

cd /home/hess/projects/discriminator-guidance/

module load anaconda/2023.09

source activate climsim

srun /home/hess/.conda/envs/climsim/bin/python3.12 search.py 
