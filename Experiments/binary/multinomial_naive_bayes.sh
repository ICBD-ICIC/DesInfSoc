#!/bin/bash
#SBATCH --job-name=itrust-multinomial_naive_bayes
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=28,30,31,32,34,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/multinomial_naive_bayes-%A-%a.out
#SBATCH --error=errors/multinomial_naive_bayes-%A-%a.err

source ../.experiments_env/bin/activate

srun python multinomial_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY
