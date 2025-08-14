#!/bin/bash

##SBATCH --qos=gpushort
##SBATCH --qos=gpulong
#SBATCH --qos=gpumedium
##SBATCH --qos=gpupreempt

#SBATCH --job-name=dm-2dv
#SBATCH --partition=gpu
#SBATCH --mem=248G

#SBATCH --gres=gpu:1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=8

#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err

cd /home/hess/projects/constrained-generation/

module load anaconda/2023.09

source activate climsim

CONFIG="diffusion_vorticity_config.yaml"
BACKUP_DIR="/home/hess/projects/constrained-generation/jobs/logged_configs/"

cp configs/$CONFIG $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG

srun /home/hess/.conda/envs/climsim/bin/python3.12 training.py -c $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG
