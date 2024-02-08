#!/bin/bash
#SBATCH --job-name=itrust-logistic_regression
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=28,30,32,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/logistic_regression-%A-%a.out
#SBATCH --error=errors/logistic_regression-%A-%a.err

source .experiments_env/bin/activate

srun python logistic_regression.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY