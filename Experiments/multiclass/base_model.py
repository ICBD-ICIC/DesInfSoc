import sys
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score)
from imblearn.under_sampling import RandomUnderSampler

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
AVERAGE_OPTIONS = ['micro', 'macro', 'weighted']

context_columns = list(range(0, 26))

prediction = sys.argv[1]
dataset_name = sys.argv[2]


def get_dataset():
    dataset = pd.read_csv('dataset/{}.csv'.format(dataset_name))
    X = dataset.iloc[:, context_columns]
    y = dataset.iloc[:, int(prediction)]
    return X, y


# Maintains class balance
def get_train_test_split():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    print('TOTAL Amount per class, original:')
    print(y.value_counts())
    print('TRAIN Amount per class, original:')
    print(y_train.value_counts())

    minority_class = min(y_train.value_counts())
    strategy = dict(y_train.value_counts())
    strategy = {class_name: int(min(amount, minority_class * 1.6)) for class_name, amount in strategy.items()}

    print(strategy)

    X_train, y_train = RandomUnderSampler(random_state=42, sampling_strategy=strategy).fit_resample(X_train, y_train)
    print('TRAIN Amount per class, after random under sampler:')
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def get_output_filepath(model_name):
    return 'experiments/{},{},{},{}'.format(dataset_name, model_name, PREDICTION_MAP[prediction], time.time())


def get_metrics(y_test, y_pred):
    metrics = {'accuracy': accuracy_score(y_test, y_pred)}
    for average_option in AVERAGE_OPTIONS:
        metrics['precision_{}'.format(average_option)] = precision_score(y_test, y_pred, average=average_option)
        metrics['recall_{}'.format(average_option)] = recall_score(y_test, y_pred, average=average_option)
        metrics['f1_{}'.format(average_option)] = f1_score(y_test, y_pred, average=average_option)
        for beta_option in BETA_OPTIONS:
            metrics['fbeta_{}_{}'.format(average_option, beta_option)] = \
                fbeta_score(y_test, y_pred, average=average_option, beta=beta_option)
    return metrics
