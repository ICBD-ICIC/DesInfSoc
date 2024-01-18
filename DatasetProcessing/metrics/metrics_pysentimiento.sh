#!/bin/bash
#SBATCH --job-name=itrust-sentiments
#SBATCH --array=0-58
#SBATCH --time=05-00:00:00
#SBATCH --output=outputs2/sentiments-%A-%a.out
#SBATCH --error=errors2/sentiments-%A-%a.err

source .sentiments_env/bin/activate

srun python metrics_pysentimiento.py ${SLURM_ARRAY_TASK_ID}

