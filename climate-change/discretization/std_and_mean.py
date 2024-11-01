# Calculates standard deviation and mean for each feature inside the specified experiment_type folder

import pandas as pd
import os
import numpy as np
import json

experiment_type = 'distance'

FOLDER_PATH = '../outputs/tweets/{}'.format(experiment_type)
OUTPUT_FILE = 'std_and_mean_{}.json'.format(experiment_type)

file_list = os.listdir(FOLDER_PATH)

stds_and_means = {}

for file in file_list:
    file_data = pd.read_csv(FOLDER_PATH + "/" + file)
    filename = file.split('.')[0]

    for col in file_data.filter(regex='.*_n').columns:
        stds_and_means['{}_{}_std'.format(filename, col)] = np.std(file_data[col])
        stds_and_means['{}_{}_mean'.format(filename, col)] = np.mean(file_data[col])

    for col in file_data.filter(regex='.*_ratio').columns:
        stds_and_means['{}_{}_std'.format(filename, col)] = np.std(file_data[col])
        stds_and_means['{}_{}_mean'.format(filename, col)] = np.mean(file_data[col])

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(stds_and_means, outfile)
