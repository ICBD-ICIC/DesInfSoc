#!/bin/bash

DATASETS=(
    "context_ONLY-ACTION-SPREAD20_K3_H4_P12",
    "context_SPREAD20_K3_H4_P12",
    "context_ONLY-ACTION-SPREAD60_K3_H4_P12",
    "context_SPREAD60_K3_H4_P12"
)

# Loop through each dataset and submit the sbatch job
for X in "${DATASETS[@]}"
do
    sbatch decision_tree.sh "$X"
    sbatch logistic_regression.sh "$X"
    sbatch naive_bayes.sh "$X"
    sbatch neural_network.sh "$X"
    sbatch random_forest.sh "$X"
done
