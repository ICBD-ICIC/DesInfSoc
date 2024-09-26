#!/bin/bash
#SBATCH --job-name=itrust-complement_naive_bayes
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=28,30,32,31,34,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/complement_naive_bayes-%A-%a.out
#SBATCH --error=errors/complement_naive_bayes-%A-%a.err

source ../.experiments_env/bin/activate

srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY
