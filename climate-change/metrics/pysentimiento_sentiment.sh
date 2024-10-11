#!/bin/bash
#SBATCH --job-name=itrust-sentiments
#SBATCH --mem=32G
#SBATCH --time=03-00:00:00
#SBATCH --output=outputs/sentiments-%A-%a.out
#SBATCH --error=errors/sentiments-%A-%a.err

source .sentiments_env/bin/activate

srun python pysentimiento_sentiments.py

