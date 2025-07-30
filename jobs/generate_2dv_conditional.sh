#!/bin/bash

##SBATCH --qos=gpushort
##SBATCH --qos=gpulong
#SBATCH --qos=gpumedium
##SBATCH --qos=gpupreempt

#SBATCH --job-name=gen_cdm_2dv

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=245G

#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err

cd /home/hess/projects/discriminator-guidance/

module load anaconda/2023.09

source activate climsim

CONFIG="vorticity_conditional.yaml"

BACKUP_DIR="/home/hess/projects/discriminator-guidance/jobs/logged_configs/"
cp configs/generate/$CONFIG $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG

srun /home/hess/.conda/envs/climsim/bin/python3.12 generate.py -c $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG
