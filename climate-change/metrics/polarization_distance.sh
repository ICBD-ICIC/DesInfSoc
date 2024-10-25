#!/bin/bash
#SBATCH --job-name=itrust-polarization
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --output=outputs/polarization-%j.out
#SBATCH --error=errors/polarization-%j.err

source .experiments_env/bin/activate

srun python polarization_distance.py
