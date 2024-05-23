#!/bin/bash
#SBATCH --job-name=itrust-context
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-21
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/context-%A-%a.out
#SBATCH --error=errors/context-%A-%a.err

source .context_env/bin/activate

srun python context-action.py ${SLURM_ARRAY_TASK_ID}