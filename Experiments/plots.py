import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO']


def prediction_display_name(prediction_original):
    display_name = (prediction_original
                    .replace('_', ' ')
                    .replace('interval', '')
                    .replace('predominant', '')
                    .replace('mfd', 'MFD'))
    return display_name


FOLDER_PATH = 'results/'

file_list = os.listdir(FOLDER_PATH)

plot_data = []

for file in file_list:
    metrics = open(FOLDER_PATH + "/" + file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0]
    metrics = metrics + '}' if metrics[-1] != '}' else metrics
    metrics = json.loads(metrics)
    model = file.split(',')[1]
    prediction: str = file.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction_display_name(prediction)}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'].isin(['precision'])]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='metric_name', kind="bar",
                edgecolor='black', legend=False)
p.set(xlabel=None, ylabel=None)
p.set_titles("{col_name}")

bars_amount = 0
colors = []

# Loop through the bars and assign hatches
for axes in p.axes.flat:
    axes.set_title(axes.get_title().replace('_', ' ').capitalize())
    bars_amount = len(axes.get_xticks())
    for i, bar in enumerate(axes.patches):
        bar.set_hatch(HATCHES[int(i / bars_amount)])
        bar_color = bar.get_facecolor()
        colors.append(bar_color) if bar_color not in colors else colors

models_names = list(map(lambda name: name.replace('_', ' ').capitalize(), dataframe['model'].unique()))

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=models_names[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=10)
plt.subplots_adjust(right=0.7)

plt.show()
