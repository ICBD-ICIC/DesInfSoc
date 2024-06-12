import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

FOLDER_PATHS = ['binary/experiments(spread20,balanced,only-action)']
METRIC = 'f1'
TITLE = 'F1 score - new approach'

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']


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
    filename = file.replace('\\', '/').split('/')[-1]
    spread = filename.split(',')[0].split('_')[1].replace('SPREAD', '')
    model = filename.split(',')[1]
    prediction = filename.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction, 'spread': spread}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='spread', kind="bar",
                edgecolor='black', legend=False)
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

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=models_names[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=10)

plt.suptitle(TITLE, size=16)
plt.subplots_adjust(top=0.9, right=0.8)
plt.show()
