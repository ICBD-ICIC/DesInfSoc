#!/bin/bash
#SBATCH --job-name=itrust-llama
#SBATCH --mem=16G
#SBATCH --nodelist=c01
#SBATCH --partition=gpu
#SBATCH --array=0-5
#SBATCH --time=00-12:00:00
#SBATCH --output=outputs/llama-%A-%a.out
#SBATCH --error=errors/llama-%A-%a.err

PREDICTIONS=(abusive polarization emotion valence sentiment mfd)

source .experiments_env/bin/activate

srun python llama.py pattern_matching "${PREDICTIONS[$SLURM_ARRAY_TASK_ID]}" false
