import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
import numpy as np

FOLDER_PATHS_BALANCED = ['../experiments/experiments/']

FOLDER_PATHS = FOLDER_PATHS_BALANCED
METRIC = 'f1'

file_list = []
for folder_path in FOLDER_PATHS:
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_list.append(file_path)

plot_data = []

for filename in file_list:
    metrics = open(filename, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0]
    metrics = metrics + '}' if metrics[-1] != '}' else metrics
    metrics = json.loads(metrics)
    model = filename.split(',')[1]
    prediction = filename.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model, 'prediction': prediction}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", kind="bar", edgecolor='black')

plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.5)
p.set(xlabel=None, ylabel=None)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.show()
