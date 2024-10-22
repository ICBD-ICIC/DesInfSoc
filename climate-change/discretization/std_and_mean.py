import pandas as pd
import os
import numpy as np
import json

FOLDER_PATH = '../outputs/tweets/pattern_matching'
OUTPUT_FILE = 'std_and_mean_pattern_matching.json'

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
