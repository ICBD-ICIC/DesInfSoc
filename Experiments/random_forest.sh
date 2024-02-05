#!/bin/bash
#SBATCH --job-name=itrust-random_forest
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --array=28,30,32,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/random_forest-%A-%a.out
#SBATCH --error=errors/random_forest-%A-%a.err

source .experiments_env/bin/activate

srun python random_forest.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12