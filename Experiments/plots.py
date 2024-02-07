import matplotlib
import seaborn
import json
import os
import pandas as pd

FOLDER_PATH = 'results/'

file_list = os.listdir(FOLDER_PATH)

plot_data = []

#TODO: ver como mostrar todas las metricas en un solo grafico

for file in file_list:
    model_data = open(FOLDER_PATH + "/" + file, "r").read()
    model_data = " ".join(model_data.split())
    model_data = model_data.split('}{', 1)[0] + '}'
    model_data = json.loads(model_data)
    model_data['model'] = file.split(',')[1]
    model_data['prediction'] = file.split(',')[2]
    plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)

barplot = seaborn.barplot(data=dataframe, x='prediction', hue="model", y="accuracy")
seaborn.move_legend(barplot, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
matplotlib.pyplot.show()
