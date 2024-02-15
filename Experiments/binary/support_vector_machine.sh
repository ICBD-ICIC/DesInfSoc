#!/bin/bash
#SBATCH --job-name=itrust-support_vector_machine
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --array=31,34
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/support_vector_machine-%A-%a.out
#SBATCH --error=errors/support_vector_machine-%A-%a.err

source .experiments_env/bin/activate

srun python support_vector_machine.py ${SLURM_ARRAY_TASK_ID} context_SPREAD60_K3_H4_P12-BINARY
