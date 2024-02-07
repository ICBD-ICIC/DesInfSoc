import seaborn
import json
import os
import pandas as pd

FOLDER_PATH = 'experiments/'

file_list = os.listdir(FOLDER_PATH)

plot_data = []

# TODO: añadir el model y el atributo asi puedo hacer plots filtrados

for file in file_list:
    model_data = open(FOLDER_PATH + "/" + file, "r").read()
    model_data = " ".join(model_data.split())
    model_data = model_data.split('}{', 1)[0] + '}'
    model_data = json.loads(model_data)
    plot_data.append(model_data)

print(pd.DataFrame(plot_data))
