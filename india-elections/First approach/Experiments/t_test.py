import itertools
import json
import os
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", None)

PARENT_FOLDER = 'binary/feature-selection/results-t_test-30_runs(spread20,balanced)/end-to-end/'
all_items = os.listdir(PARENT_FOLDER)
experiment_subfolders = [PARENT_FOLDER + item for item in all_items if item.startswith('experiments')]
file_list = []

for folder_path in experiment_subfolders:
    file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename), os.listdir(folder_path)))

results_data = []

for file in file_list:
    k_metrics = open(file, "r").read()
    k_metrics = json.loads(k_metrics)
    # spread?
    features = file.split('/')[4].replace('experiments-', '')
    prediction = file.split('/')[5].split(',')[2]
    f1_values = []
    recall_values = []
    precision_values = []
    for instance in k_metrics:
        f1_values.append(instance['f1'])
        recall_values.append(instance['recall'])
        precision_values.append(instance['precision'])
    model_data = {'f1': f1_values, 'recall': recall_values, 'precision': precision_values,
                  'prediction': prediction, 'feature': features}
    results_data.append(model_data)

dataframe = pd.DataFrame(results_data)
features = dataframe['feature'].unique()
predictions = dataframe['prediction'].unique()

all_comparisons = list(itertools.combinations(features, 2))
query = 'prediction == "{}" & feature == "{}"'
metrics_to_compare = ['f1', 'precision', 'recall']

t_test_results = []

for metric in ['f1', 'precision', 'recall']:
    for prediction in predictions:
        for feature1, feature2 in all_comparisons:
            groups1 = set(feature1.split('+'))
            groups2 = set(feature2.split('+'))
            if groups1.issubset(groups2) or groups2.issubset(groups1):
                data_to_compare1 = dataframe.query(query.format(prediction, feature1))[metric].item()
                data_to_compare2 = dataframe.query(query.format(prediction, feature2))[metric].item()
                t_test = stats.ttest_ind(data_to_compare1, data_to_compare2, random_state=42)
                t_test_results.append({'metric': metric, 'prediction': prediction, 'feature1': feature1,
                                       'feature2': feature2, 't_test statistic': t_test.statistic,
                                       't_test pvalue': t_test.pvalue})
                # if t_test.pvalue > 0.05:
                #     print(metric, prediction, feature1, feature2, t_test)

# Save results
results_df = pd.DataFrame(t_test_results)
results_df.to_csv(PARENT_FOLDER + 't_test_results.csv', index=False)
