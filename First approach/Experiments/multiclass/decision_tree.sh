#!/bin/bash
#SBATCH --job-name=itrust-decision_tree
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=28,30,31,32,34,36
#SBATCH --time=00-01:00:00
#SBATCH --output=outputs/decision_tree-%A-%a.out
#SBATCH --error=errors/decision_tree-%A-%a.err

dataset="$1"

source ../.experiments_env/bin/activate

srun python decision_tree.py ${SLURM_ARRAY_TASK_ID} "$dataset"