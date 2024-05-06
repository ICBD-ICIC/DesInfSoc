#!/bin/bash

# List of X values
X_VALUES=(
    "emotion"
    "emotion+linguistic-ratio"
    "emotion+linguistic-ratio+personality"
    "emotion+linguistic-ratio+personality+sentiment"
    "emotion+linguistic-ratio+sentiment"
    "emotion+personality"
    "emotion+personality+sentiment"
    "emotion+sentiment"
    "linguistic-ratio"
    "linguistic-ratio+personality"
    "linguistic-ratio+personality+sentiment"
    "emotion+linguistic-ratio+personality+sentiment+linguistic-amount"
    "linguistic-ratio+sentiment"
    "personality"
    "personality+sentiment"
    "sentiment"
)

# Loop through each X value and submit the sbatch job
for X in "${X_VALUES[@]}"
do
    sbatch random_forest.sh "$X"
done
