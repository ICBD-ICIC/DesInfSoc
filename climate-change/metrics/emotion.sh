#!/bin/bash
#SBATCH --job-name=itrust-emotions
#SBATCH --mem=32G
#SBATCH --time=03-00:00:00
#SBATCH --output=../outputs/emotions-%A-%a.out
#SBATCH --error=../errors/emotions-%A-%a.err

source .sentiments_env/bin/activate

srun python emotions.py

