#!/bin/bash
#SBATCH --job-name=itrust-valence_negative
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/valence_negative-%j.out
#SBATCH --error=errors/valence_negative-%j.err

source .experiments_env/bin/activate

srun python valence_negative_distance.py
