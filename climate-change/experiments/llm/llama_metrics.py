import json
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, auc,
                             precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix)

BETA_OPTIONS = [0.5, 2, 3, 4]
AVERAGE_OPTIONS = ['micro', 'macro', 'weighted']


def precision_recall_auc(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


def multiclass_metrics(y_true, y_pred):
    metrics = {'accuracy': accuracy_score(y_true, y_pred)}
    for average_option in AVERAGE_OPTIONS:
        metrics['precision_{}'.format(average_option)] = precision_score(y_true, y_pred, average=average_option)
        metrics['recall_{}'.format(average_option)] = recall_score(y_true, y_pred, average=average_option)
        metrics['f1_{}'.format(average_option)] = f1_score(y_true, y_pred, average=average_option)
        for beta_option in BETA_OPTIONS:
            metrics['fbeta_{}_{}'.format(average_option, beta_option)] = \
                fbeta_score(y_true, y_pred, average=average_option, beta=beta_option)
    return metrics


def get_metrics(y_test, y_pred):
    if len(y_test.value_counts().index) == 2:
        return binary_metrics(y_test, y_pred)
    else:
        return multiclass_metrics(y_test, y_pred)


def binary_metrics(y_test, y_pred):
    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred),
               'f1': f1_score(y_test, y_pred),
               'roc_auc': roc_auc_score(y_test, y_pred),
               'precision_recall_auc': precision_recall_auc(y_test, y_pred)}
    for beta_option in BETA_OPTIONS:
        metrics['fbeta_{}'.format(beta_option)] = fbeta_score(y_test, y_pred, beta=beta_option)
    return metrics


def save_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    plt.savefig(f'confusion_matrix/{filename.replace('csv','png')}')


LABEL_MAPPING = {'Low': 0,
                 'High': 1,
                 'Neutral': 0,
                 'Anger': 1,
                 'Disgust': 2,
                 'Fear': 3,
                 'Joy': 4,
                 'Sadness': 5,
                 'Surprise': 6,
                 'Equal': 0,
                 'Virtue': 1,
                 'Vice': 2,
                 'Positive': 1,
                 'Negative': 2}
RESULTS_FOLDER = "results"

for filename in os.listdir(RESULTS_FOLDER):
    df = pd.read_csv(os.path.join(RESULTS_FOLDER, filename))
    df['result'] = df['result'].str.replace(r'More |\.', '', regex=True)
    df['ground_truth'] = df['ground_truth'].map(LABEL_MAPPING)
    df['result'] = df['result'].map(LABEL_MAPPING)
    metrics = get_metrics(df['ground_truth'], df['result'])

    with open(f'metrics/{filename}', 'w') as file:
        file.write(json.dumps(metrics, indent=4))

    save_confusion_matrix(df['ground_truth'], df['result'])
