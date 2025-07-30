#!/bin/bash

##SBATCH --qos=gpushort
##SBATCH --qos=gpulong
##SBATCH --qos=gpumedium
#SBATCH --qos=gpupreempt

##SBATCH --job-name=dm-2dv
#SBATCH --job-name=cdm-nsub-2dv
#SBATCH --partition=gpu
#SBATCH --mem=248G

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err

cd /home/hess/projects/discriminator-guidance/

module load anaconda/2023.09

source activate climsim

#CONFIG="diffusion_vorticity_config.yaml"
#CONFIG="diffusion_cond_vorticity_config.yaml"
CONFIG="diffusion_cond_vorticity_nonsub_config.yaml"

BACKUP_DIR="/home/hess/projects/discriminator-guidance/jobs/logged_configs/"
cp configs/$CONFIG $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG

srun /home/hess/.conda/envs/climsim/bin/python3.12 diffusion_training.py -c $BACKUP_DIR/${SLURM_JOB_ID}_$CONFIG
#srun /home/hess/.conda/envs/climsim/bin/python3.12 diffusion_training.py -c diffusion_config.yaml
