import os
import pandas as pd

PREDICTION_MAP = {
    '28': 'abusive_ratio_interval',
    '30': 'polarization_ratio_interval',
    '31': 'predominant_emotion',
    '32': 'mfd_ratio',
    '34': 'valence_ratio',
    '36': 'predominant_sentiment'
}

DATASETS_FOLDER = 'dataset/'

# dataset_list = os.listdir(DATASETS_FOLDER)
dataset_list = ['context_SPREAD20_K3_H4_P12.csv']
datasets_info = []

for dataset_path in dataset_list:
    dataset = pd.read_csv(DATASETS_FOLDER + dataset_path)
    for prediction_number, prediction_name in PREDICTION_MAP.items():
        print(prediction_name)
        y = dataset.iloc[:, int(prediction_number)]
        print(y.value_counts())

#
# spread
# class_balance
# classification_type
# approach