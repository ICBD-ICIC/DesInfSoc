#!/bin/bash
#SBATCH --job-name=itrust-emotions
#SBATCH --array=0-14
#SBATCH --time=05-00:00:00
#SBATCH --output=outputs/emotions-%A-%a.out
#SBATCH --error=errors/emotions-%A-%a.err

source .emotions_env/bin/activate

srun python metric_calculator_emotions.py ${SLURM_ARRAY_TASK_ID} 
