#!/bin/bash
#SBATCH --job-name=itrust-valence_positive
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00:00
#SBATCH --output=outputs/valence_positive-%j.out
#SBATCH --error=errors/valence_positive-%j.err

source .experiments_env/bin/activate

srun python valence_positive_distance.py
