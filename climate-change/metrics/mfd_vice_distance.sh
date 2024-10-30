#!/bin/bash
#SBATCH --job-name=itrust-mfd_vice
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/mfd_vice-%j.out
#SBATCH --error=errors/mfd_vice-%j.err

source .experiments_env/bin/activate

srun python mfd_vice_distance.py
