import os
import pandas as pd

context_predictions = [28, 30, 31, 32, 34, 36]
context2_predictions = [34, 37, 39, 41, 45, 48]

FOLDER_PATH = 'dataset\contexts'

file_list = [os.path.join(FOLDER_PATH, file) for file in os.listdir(FOLDER_PATH)]

data = []

for file in file_list:
    df = pd.read_csv(file)
    predictions = context_predictions
    words_present = False
    if 'context2' in file:
        predictions = context2_predictions
        words_present = True
    spread = file.split('_')[1].split('-')[-1].replace('SPREAD', '')
    if 'ONLY-ACTION' in file:
        approach = 'only-action'
    else:
        approach = 'end-to-end'
    for prediction in predictions:
        data.append({'prediction': prediction, 'spread': spread, 'approach': approach,
                     'words_present': words_present, **df.iloc[:, prediction].value_counts().to_dict()})

data_df = pd.DataFrame(data)
data_df.to_csv('class_amount.csv')
