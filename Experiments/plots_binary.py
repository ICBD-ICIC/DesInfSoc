import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

FOLDER_PATHS = ['binary/comparison']
METRIC = 'f1'
TITLE = 'F1 score - first approach'

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
    spread = file.split(',')[0].split('_')[1].replace('SPREAD', '')
    model = file.split(',')[1]
    prediction = file.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction, 'spread': spread}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='spread', kind="bar",
                edgecolor='black')
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

# plt.tight_layout()
plt.suptitle(TITLE, size=16)
plt.subplots_adjust(top=0.9)
plt.show()
