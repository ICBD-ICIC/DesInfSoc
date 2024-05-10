from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

FOLDER_PATHS = ['binary/feature-selection/results-t_test-30_runs(spread20,balanced)/only-action/',
                'binary/feature-selection/results-t_test-30_runs(spread20,balanced)/end-to-end/']
METRIC = 'recall'
TITLE = 'Recall - average of 30 runs'

HATCHES = ['\\\\', '-', '//', '..', '', '++', 'oo', '||', 'XX', 'OO', '\\', '--', '/', '.', '+', 'o', '|', 'X', 'o']

def paper_experiment_one(dataframe):
    features = {'emotion': 'Only emotion',
                'emotion+linguistic-ratio': 'Emotion + Linguistic Ratio',
                'personality': 'Only personality',
                'linguistic-ratio+personality': 'Personality + Linguistic Ratio',
                'sentiment': 'Only sentiment',
                'linguistic-ratio+sentiment': 'Sentiment + Linguistic Ratio',
                'emotion+personality': 'Emotion + Personality',
                'emotion+linguistic-ratio+personality': 'Emotion + Personality + Linguistic Ratio',
                'emotion+sentiment': 'Emotion + Sentiment',
                'emotion+linguistic-ratio+sentiment': 'Emotion + Sentiment + Linguistic Ratio',
                'personality+sentiment': 'Personality + Sentiment',
                'linguistic-ratio+personality+sentiment': 'Personality + Sentiment + Linguistic Ratio',
                'emotion+personality+sentiment': 'Emotion + Sentiment + Personality',
                'emotion+linguistic-ratio+personality+sentiment': 'Emotion + Sentiment + Personality + Linguistic Ratio',
                'emotion+linguistic-ratio+personality+sentiment+linguistic-amount': 'All'
                }
    df = dataframe.copy()
    df = df.loc[dataframe['feature'].isin(features.keys())]
    df['feature'] = pd.Categorical(df['feature'], categories=features.keys(), ordered=True)
    df['feature'] = df['feature'].replace(features)

    predictions = {'predominant_emotion': 'P5',
                   'predominant_sentiment': 'P4',
                   'valence_ratio': 'P6',
                   'abusive_ratio_interval': 'P1',
                   'mfd_ratio': 'P3',
                   'polarization_ratio_interval': 'P2'}
    df['prediction'] = df['prediction'].replace(predictions)

    df = df.sort_values(by=['feature', 'prediction'])

    df['experiment_type'] = df['experiment_type'].str.replace('-', ' ').str.capitalize()

    return df


plot_data = []

experiment_subfolders = []
for folder_path in FOLDER_PATHS:
    all_items = os.listdir(folder_path)
    experiment_subfolders.extend([folder_path + item for item in all_items if item.startswith('experiments')])

file_list = []

for folder_path in experiment_subfolders:
    file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename), os.listdir(folder_path)))

for file in file_list:
    k_metrics = open(file, "r").read()
    k_metrics = json.loads(k_metrics)
    feature = file.split('/')[4].replace('experiments-', '')
    prediction = file.split('/')[5].split(',')[2]
    f1_values = []
    recall_values = []
    precision_values = []
    experiment_type = file.split('/')[3]
    for instance in k_metrics:
        f1_values.append(instance['f1'])
        recall_values.append(instance['recall'])
        precision_values.append(instance['precision'])
    plot_data.append({'metric_name': 'f1', 'metric_value': mean(f1_values), 'prediction': prediction,
                      'feature': feature, 'experiment_type': experiment_type})
    plot_data.append({'metric_name': 'recall', 'metric_value': mean(recall_values), 'prediction': prediction,
                      'feature': feature, 'experiment_type': experiment_type})
    plot_data.append({'metric_name': 'precision', 'metric_value': mean(precision_values), 'prediction': prediction,
                      'feature': feature, 'experiment_type': experiment_type})

dataframe = pd.DataFrame(plot_data)
dataframe = paper_experiment_one(dataframe)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="feature", kind="bar", col='experiment_type',
                edgecolor='black', legend=False)
p.set(xlabel=None, ylabel=None)

colors = []
features = dataframe['feature'].unique()

# Loop through the bars and assign hatches
for axes in p.axes.flat:
    axes.set_title(axes.get_title().replace('_', ' ').capitalize())
    for bar in axes.patches:
        bar_color = bar.get_facecolor()
        colors.append(bar_color) if bar_color not in colors else colors
        bar.set_hatch(HATCHES[colors.index(bar_color)])
    # shows label with value of each bar
    # for i in axes.containers:
    #     axes.bar_label(i, )

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=features[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches,  borderaxespad=0, fontsize=10,
           ncols=4, loc='lower center', bbox_to_anchor=(-1, -0.25, 2, 1))
plt.suptitle(TITLE, size=16)
plt.subplots_adjust(top=0.9, bottom=0.2)
plt.show()
