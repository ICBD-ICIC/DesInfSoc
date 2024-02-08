import matplotlib
import seaborn as sns
import json
import os
import pandas as pd

FOLDER_PATH = 'results/'

file_list = os.listdir(FOLDER_PATH)

plot_data = []


#TODO: ver como mostrar todas las metricas en un solo grafico

for file in file_list:
    metrics = open(FOLDER_PATH + "/" + file, "r").read()
    metrics = " ".join(metrics.split())
    metrics = metrics.split('}{', 1)[0] + '}'
    metrics = json.loads(metrics)
    model = file.split(',')[1]
    prediction = file.split(',')[2]
    print(metrics)
    for name, value in metrics.items():
        model_data = {'metric_name': name, 'metric_value': value, 'model': model, 'prediction': prediction}
        plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe = dataframe.loc[dataframe['metric_name'].isin(['accuracy', 'precision_macro', 'recall_macro'])]

print(dataframe['metric_name'].value_counts())

grid = sns.FacetGrid(dataframe, col="metric_name")


grid.map_dataframe(sns.barplot, 'prediction', 'metric_value', 'model')
grid.add_legend()
matplotlib.pyplot.show()
