#!/bin/bash
#SBATCH --job-name=itrust-polarization
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --output=outputs/polarization-%A-%a.out
#SBATCH --error=errors/polarization-%A-%a.err

source ../.experiments_env/bin/activate

srun python polarization_distance.py
