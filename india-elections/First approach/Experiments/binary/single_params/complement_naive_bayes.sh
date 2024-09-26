#!/bin/bash
#SBATCH --job-name=itrust-complement_naive_bayes
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=28
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/complement_naive_bayes-%A-%a.out
#SBATCH --error=errors/complement_naive_bayes-%A-%a.err

source .experiments_env/bin/activate

srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.05 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.1 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.15 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.2 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 10 True
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0 False
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.05 False
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.1 False
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.15 False
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 0.2 False
srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY 10 False
