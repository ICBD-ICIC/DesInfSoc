import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

FOLDER_PATH = 'multiclass/results-all-hyperparameters-balanced/'
METRIC = 'f1_weighted'

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']

def prediction_display_name(prediction_original):
    display_name = (prediction_original
                    .replace('_', ' ')
                    .replace('interval', '')
                    .replace('predominant', '')
                    .replace('mfd', 'MFD'))
    return display_name


file_list = os.listdir(FOLDER_PATH)

plot_data = []

for file in file_list:
    metrics = open(FOLDER_PATH + "/" + file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0]
    metrics = metrics + '}' if metrics[-1] != '}' else metrics
    metrics = json.loads(metrics)
    spread = file.split(',')[0].split('_')[1].replace('SPREAD','')
    model = file.split(',')[1]
    prediction: str = file.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction_display_name(prediction), 'spread': spread}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='spread', kind="bar",
                edgecolor='black', legend=False)
p.set(xlabel=None, ylabel=None)
p.set_titles("{col_name}")

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
plt.legend(handles=legend_patches, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=10)
plt.subplots_adjust(right=0.8)

plt.show()
