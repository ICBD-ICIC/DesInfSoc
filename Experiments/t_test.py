import json
import os
import pandas as pd
from scipy import stats
import numpy as np

FOLDERS_PATHS = ['binary/results-all',
                 'binary/results-no-personality',
                 'binary/results-no-linguistic',
                 'binary/results-no-emotions-no-sentiments']

file_list = []

for folder_path in FOLDERS_PATHS:
    file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename),os.listdir(folder_path)))

results_data = []

for file in file_list:
    metrics = open(file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0]
    metrics = metrics + '}' if metrics[-1] != '}' else metrics
    metrics = json.loads(metrics)
    model = file.split(',')[1]
    ablation = '-'.join(file.split('/')[1].split('-')[1:])
    prediction = file.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'ablation': ablation, 'prediction': prediction}
        results_data.append(model_data)

dataframe = pd.DataFrame(results_data)
dataframe = dataframe.loc[dataframe['metric_name'] == 'recall']
ablations_to_compare = list(set(dataframe['ablation'].unique()) - {'all'})
predictions = dataframe['prediction'].unique()
models = dataframe['model'].unique()

for prediction in predictions:
    print('PREDICTION: {}'.format(prediction))
    all = dataframe.loc[(dataframe['prediction'] == prediction) & (dataframe['ablation'] == 'all')]
    for ablation in ablations_to_compare:
        print('  All vs. {}'.format(ablation))
        comparison = dataframe.loc[(dataframe['prediction'] == prediction) & (dataframe['ablation'] == ablation)]
        for model in models:
            print('    model: {}'.format(model))
            all_metrics = all.loc[all['model'] == model]['metric_value']
            comparison_metrics = comparison.loc[comparison['model'] == model]['metric_value']
            print(all_metrics)
            print(comparison_metrics)
            t_test = stats.ttest_ind(all_metrics, comparison_metrics)
            print('    {}'.format(t_test))

