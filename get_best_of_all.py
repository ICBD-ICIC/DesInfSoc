import json
import os

import pandas as pd

BINARY_FOLDERS = [
    'Experiments/binary/experiments(spread20,balanced,only-action)',
    'Experiments/binary/experiments(spread100,imbalanced,only-action)',
    'First approach/Experiments/binary/results-all-hyperparameters-balanced(end-to-end)',
    'First approach/Experiments/binary/results-all-hyperparameters-balanced(only-action)',
    'First approach/Experiments/binary/results-all-hyperparameters-imbalanced(end-to-end)',
    'First approach/Experiments/binary/results-all-hyperparameters-imbalanced(only-action)',
]
MULTICLASS_FOLDERS = [
    'First approach/Experiments/multiclass/results-all-hyperparameters-balanced(end-to-end)',
    'First approach/Experiments/multiclass/results-all-hyperparameters-balanced(only-action)'
]


def get_all_results_files(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            file_list.append(file_path)
    return file_list


def get_metrics_values(file_list, metrics_list):
    metrics_values = []

    for file in file_list:
        metrics = open(file, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)
        prediction = file.replace('\\', '/').split('/')[-1].split(',')[2]
        metrics = {key: metrics[key] for key in metrics_list}
        for name, value in metrics.items():
            model_data = {f"{name}": value, 'file': file, 'prediction': prediction}
            metrics_values.append(model_data)

    return pd.DataFrame(metrics_values)


# BINARY #
print('BINARY')
df = get_metrics_values(get_all_results_files(BINARY_FOLDERS), ['f1'])
for prediction in df['prediction'].unique():
    best_result_idx = df.loc[df['prediction'] == prediction]['f1'].idxmax()
    print('Best f1 score for', prediction, 'is on file', df.iloc[best_result_idx]['file'])

# MULTICLASS #
print('MULTICLASS')
df = get_metrics_values(get_all_results_files(MULTICLASS_FOLDERS), ['f1_micro', 'f1_macro', 'f1_weighted'])
for prediction in df['prediction'].unique():
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_micro'].idxmax()
    print('Best f1_micro score for', prediction, 'is on file', df.iloc[best_result_idx]['file'])
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_macro'].idxmax()
    print('Best f1_macro score for', prediction, 'is on file', df.iloc[best_result_idx]['file'])
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_weighted'].idxmax()
    print('Best f1_weighted score for', prediction, 'is on file', df.iloc[best_result_idx]['file'])
