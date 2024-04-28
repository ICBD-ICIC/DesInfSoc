#!/bin/bash
#SBATCH --job-name=itrust-naive_bayes
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=31
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/naive_bayes-%A-%a.out
#SBATCH --error=errors/naive_bayes-%A-%a.err

source ../.experiments_env/bin/activate

srun python naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_ONLY-ACTION-SPREAD20_K3_H4_P12