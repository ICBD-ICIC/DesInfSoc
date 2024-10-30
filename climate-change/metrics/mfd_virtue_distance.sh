#!/bin/bash
#SBATCH --job-name=itrust-mfd_virtue
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/mfd_virtue-%j.out
#SBATCH --error=errors/mfd_virtue-%j.err

source .experiments_env/bin/activate

srun python mfd_virtue_distance.py
