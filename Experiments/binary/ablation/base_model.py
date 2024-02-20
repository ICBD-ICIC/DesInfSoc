import sys
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, auc,
                             precision_recall_curve)
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
ABLATION_FEATURES_TO_REMOVE = {
    'no-personality': [0, 1],
    'no-linguistic': [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21],
    'no-emotions-no-sentiments': [6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25],
    'no-bif-five': [0],
    'no-symanto-psychographics': [1],
    'no-linguistic-amount': [2, 4, 14, 16, 18, 20],
    'all': []
}

BETA_OPTIONS = [0.5, 2, 3, 4]


def get_context_columns(ablation_type):
    all_context = list(range(0, 26))
    context_to_remove = ABLATION_FEATURES_TO_REMOVE[ablation_type]
    return list(set(all_context) - set(context_to_remove))


prediction = sys.argv[1]
dataset_name = sys.argv[2]
ablation_type = sys.argv[3]

context_columns = get_context_columns(ablation_type)


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
    X_train, y_train = RandomUnderSampler(random_state=42, sampling_strategy=1).fit_resample(X_train, y_train)
    print('TRAIN Amount per class, after random under sampler:')
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def get_output_filepath(model_name):
    return 'experiments-{}/{},{},{},{}'.format(ablation_type, dataset_name, model_name, PREDICTION_MAP[prediction],
                                               time.time())


def precision_recall_auc(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


def get_metrics(y_test, y_pred):
    metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred),
               'roc_auc': roc_auc_score(y_test, y_pred),
               'precision_recall_auc': precision_recall_auc(y_test, y_pred)}
    for beta_option in BETA_OPTIONS:
        metrics['fbeta_{}'.format(beta_option)] = fbeta_score(y_test, y_pred, beta=beta_option)
    return metrics
