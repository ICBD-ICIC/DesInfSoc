import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO']

PREDICTION = 'polarization_ratio'
METRICS = ['precision','recall']

MODELS_SHORTNAMES = {
    'complement_naive_bayes': 'CNB',
    'decision_tree': 'DT',
    'logistic_regression': 'LG',
    'multinomial_naive_bayes': 'MNB',
    'neural_network': 'NN',
    'random_forest': 'RF',
    'random_guessing': 'RG'
}

FOLDERS_PATHS = ['binary/results-all',
                 # 'binary/results-no-personality',
                 # 'binary/results-no-linguistic',
                 # 'binary/results-no-emotions-no-sentiments',
                 # 'binary/results-no-linguistic-amount',
                 # 'binary/results-no-linguistic-amount-no-personality',
                 'binary/results-only-linguistic-ratio',
                 'binary/results-only-personality',
                 'binary/results-only-emotions-and-sentiments',
                 ]


file_list = []

for folder_path in FOLDERS_PATHS:
    file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename),os.listdir(folder_path)))

plot_data = []

for file in file_list:
    if PREDICTION in file:
        metrics = open(file, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)
        model = file.split(',')[1]
        ablation = '-'.join(file.split('/')[1].split('-')[1:])
        for name, value in metrics.items():
            model_data = {'metric_name': name, 'metric_value': value, 'model': MODELS_SHORTNAMES[model],
                          'ablation': ablation}
            plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'].isin(METRICS)]

sns.set_style("whitegrid")

p = sns.catplot(data=dataframe, x="model", y="metric_value", hue="ablation", col='metric_name', kind="bar",
                edgecolor='black', legend=False)
p.set(xlabel=None, ylabel=None)
p.set_titles("{col_name}")

colors = []
ablations = list(map(lambda name: name.replace('-', ' ').capitalize(), dataframe['ablation'].unique()))

# Loop through the bars and assign hatches
for axes in p.axes.flat:
    axes.set_title(axes.get_title().replace('_', ' ').capitalize())
    for bar in axes.patches:
        bar_color = bar.get_facecolor()
        colors.append(bar_color) if bar_color not in colors else colors
        bar.set_hatch(HATCHES[colors.index(bar_color)])

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=ablations[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=10)
plt.subplots_adjust(right=0.8)

plt.show()
