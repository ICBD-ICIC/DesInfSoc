import sys
import pandas as pd
import time

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, auc,
                             precision_recall_curve)

KFOLD = 3
TEST_SIZE = 0.1
PREDICTION_MAP = {
    '27': 'abusive_amount_interval',
    '28': 'abusive_ratio_interval',
    '29': 'polarization_amount_interval',
    '30': 'polarization_ratio_interval',
    '31': 'predominant_emotion',
    '32': 'mfd_ratio',
    '33': 'mfd_amount',
    '34': 'valence_ratio',
    '35': 'valence_amount',
    '36': 'predominant_sentiment'
}

BETA_OPTIONS = [0.5, 2, 3, 4]

context_columns = list(range(0, 26))

prediction = sys.argv[1]
dataset_name = sys.argv[2]


def precision_recall_auc(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


def get_dataset():
    dataset = pd.read_csv('dataset/{}.csv'.format(dataset_name))
    X = dataset.iloc[:, context_columns]
    y = dataset.iloc[:, int(prediction)]
    return X, y


def get_train_test_split():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    print('TOTAL Amount per class, original:')
    print(y.value_counts())
    print('TRAIN Amount per class, original:')
    print(y_train.value_counts())
    X_train, y_train = RandomUnderSampler(random_state=42, sampling_strategy=1).fit_resample(X_train, y_train)
    print('TRAIN Amount per class, after random under sampler:')
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def get_output_filepath(model_name):
    return 'experiments/{},{},{},{}'.format(dataset_name, model_name, PREDICTION_MAP[prediction], time.time())


def get_metrics(y_test, y_pred):
    metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred),
               'roc_auc': roc_auc_score(y_test, y_pred),
               'precision_recall_auc': precision_recall_auc(y_test, y_pred)}
    for beta_option in BETA_OPTIONS:
        metrics['fbeta_{}'.format(beta_option)] = fbeta_score(y_test, y_pred, beta=beta_option)
    return metrics
