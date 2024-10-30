#!/bin/bash
#SBATCH --job-name=itrust-abusive
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --output=outputs/abusive-%j.out
#SBATCH --error=errors/abusive-%j.err

source .experiments_env/bin/activate

srun python abusive_distance.py
