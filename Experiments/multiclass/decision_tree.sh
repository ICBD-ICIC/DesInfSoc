#!/bin/bash
#SBATCH --job-name=itrust-decision_tree
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=31
#SBATCH --time=00-12:00:00
#SBATCH --output=outputs/decision_tree-%A-%a.out
#SBATCH --error=errors/decision_tree-%A-%a.err

source ../.experiments_env/bin/activate

srun python decision_tree.py ${SLURM_ARRAY_TASK_ID} context_ONLY-ACTION-SPREAD20_K3_H4_P12