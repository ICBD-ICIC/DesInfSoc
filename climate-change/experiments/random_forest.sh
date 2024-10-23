#!/bin/bash
#SBATCH --job-name=itrust-random_forest
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --time=00-01:00:00
#SBATCH --output=outputs/random_forest-%A-%a.out
#SBATCH --error=errors/random_forest-%A-%a.err

PREDICTIONS=(abusive_ratio_interval_gt polarization_ratio_interval_gt predominant_emotion_gt mfd_ratio_gt valence_ratio_gt predominant_sentiment_gt)

source .experiments_env/bin/activate

srun python random_forest.py "${PREDICTIONS[$SLURM_ARRAY_TASK_ID]}" CONTEXT_pattern_matching-binary true 1