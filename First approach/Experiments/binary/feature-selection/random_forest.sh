#!/bin/bash
#SBATCH --job-name=itrust-random_forest
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --array=28,30,31,32,34,36
#SBATCH --time=00-00:10:00
#SBATCH --output=outputs/random_forest-%A-%a.out
#SBATCH --error=errors/random_forest-%A-%a.err

features="$1"

source ../../.experiments_env/bin/activate

srun python random_forest.py ${SLURM_ARRAY_TASK_ID} context_SPREAD20_K3_H4_P12-BINARY "$features"