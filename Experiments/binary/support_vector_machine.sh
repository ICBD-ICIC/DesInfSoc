#!/bin/bash
#SBATCH --job-name=itrust-support_vector_machine
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=28,30,31,32,34,36
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/support_vector_machine-%A-%a.out
#SBATCH --error=errors/support_vector_machine-%A-%a.err

source ../.experiments_env/bin/activate

srun python support_vector_machine.py ${SLURM_ARRAY_TASK_ID} context_ONLY-ACTION_K3_H4_P12-BINARY
