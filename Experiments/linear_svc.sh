#!/bin/bash
#SBATCH --job-name=itrust-linear_svc
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=28,30,32,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/linear_svc-%A-%a.out
#SBATCH --error=errors/linear_svc-%A-%a.err

source .experiments_env/bin/activate

srun python linear_svc.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12