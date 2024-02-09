import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd

FOLDER_PATH = 'results/'

file_list = os.listdir(FOLDER_PATH)

plot_data = []

for file in file_list:
    metrics = open(FOLDER_PATH + "/" + file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0] + '}'
    metrics = json.loads(metrics)
    model = file.split(',')[1]
    prediction = file.split(',')[2]
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model, 'prediction': prediction.split('_')[0]}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'].isin(['accuracy', 'precision_macro', 'recall_macro'])]

sns.set_style("whitegrid")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="model", col='metric_name', kind="bar")
p.set(xlabel=None, ylabel=None)
p.set_titles("{col_name}")

plt.show()
