import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

FOLDER_PATHS = ['multiclass/results-all-hyperparameters-balanced(only-action)/',
                'multiclass/results-all-hyperparameters-balanced(end-to-end)/']
METRIC = 'precision_macro'
TITLE = 'Precision - macro'

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']


def paper_experiments(dataframe):
    predictions = {'predominant_emotion': 'P5',
                   'predominant_sentiment': 'P4',
                   'valence_ratio': 'P6',
                   'abusive_ratio_interval': 'P1',
                   'mfd_ratio': 'P3',
                   'polarization_ratio_interval': 'P2'}
    df = dataframe
    df['prediction'] = df['prediction'].replace(predictions)
    df['spread'] = df['spread'].str[-2:]
    df['experiment_type'] = df['experiment_type'].str.replace('-', ' ').str.capitalize()
    df = df.sort_values(by=['spread', 'prediction'])
    return df


file_list = []
for folder_path in FOLDER_PATHS:
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_list.append(file_path)

plot_data = []

for file in file_list:
    metrics = open(file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0]
    metrics = metrics + '}' if metrics[-1] != '}' else metrics
    metrics = json.loads(metrics)
    spread = file.split(',')[0].split('_')[1].replace('SPREAD', '')
    model = file.split(',')[1]
    prediction = file.split(',')[2]
    experiment_type = file.split('(')[1].split(')')[0]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction, 'spread': spread, 'experiment_type': experiment_type}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = paper_experiments(dataframe)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='spread', kind="bar",
                edgecolor='black', row="experiment_type", legend=False)
p.set(xlabel=None, ylabel=None)

colors = []
models_names = list(map(lambda name: name.replace('_', ' ').capitalize(), dataframe['model'].unique()))

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

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=models_names[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches, borderaxespad=0, fontsize=10, mode='expand',
           ncol=len(models_names), loc='lower center', bbox_to_anchor=(-1, -0.2, 2, 1))
#plt.tight_layout()
plt.suptitle(TITLE, size=16)
plt.subplots_adjust(top=0.9, bottom=0.1)
plt.show()
