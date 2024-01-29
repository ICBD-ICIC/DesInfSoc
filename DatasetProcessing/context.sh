#!/bin/bash
#SBATCH --job-name=itrust-context
#SBATCH --array=6
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/context-%A-%a.out
#SBATCH --error=errors/context-%A-%a.err

source .context_env/bin/activate

srun python context.py ${SLURM_ARRAY_TASK_ID} 

