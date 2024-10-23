import sys
import pandas as pd
import time

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, auc,
                             precision_recall_curve)

KFOLD = 3
TEST_SIZE = 0.1
BETA_OPTIONS = [0.5, 2, 3, 4]

CONTEXT_FEATURES = ['big_five',
                    'psychographics',
                    'abusive_amount_interval',
                    'abusive_ratio_interval',
                    'polarization_amount_interval',
                    'polarization_ratio_interval',
                    'predominant_emotion',
                    'neutral',
                    'anger',
                    'disgust',
                    'fear',
                    'joy',
                    'sadness',
                    'surprise',
                    'mfd_virtue_amount',
                    'mfd_virtue_ratio',
                    'mfd_vice_amount',
                    'mfd_vice_ratio',
                    'valence_positive_amount',
                    'valence_positive_ratio',
                    'valence_negative_amount',
                    'valence_negative_ratio',
                    'predominant_sentiment',
                    'sentiment-neutral',
                    'sentiment-positive',
                    'sentiment-negative']

prediction = sys.argv[1]
dataset_name = sys.argv[2]
if len(sys.argv) >= 5:
    balance = sys.argv[3].lower() == 'true'
    sampling_strategy = float(sys.argv[4])
else:
    balance = False

def precision_recall_auc(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


def get_dataset():
    dataset = pd.read_csv('dataset/{}.csv'.format(dataset_name))
    X = dataset[CONTEXT_FEATURES]
    y = dataset[prediction]
    return X, y


def get_train_test_split():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    if balance:
        print('TOTAL Amount per class, original:')
        print(y.value_counts())
        print('TRAIN Amount per class, original:')
        print(y_train.value_counts())
        X_train, y_train = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy).fit_resample(X_train, y_train)
        print('TRAIN Amount per class, after random under sampler:')
        print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def get_output_filepath(model_name):
    output_filename = 'experiments/{},{},{}'.format(dataset_name, model_name, prediction)
    if balance:
        output_filename += ',balanced,{}'.format(sampling_strategy)
    output_filename += ',{}'.format(time.time())
    return output_filename


def get_metrics(y_test, y_pred):
    metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred),
               'roc_auc': roc_auc_score(y_test, y_pred),
               'precision_recall_auc': precision_recall_auc(y_test, y_pred)}
    for beta_option in BETA_OPTIONS:
        metrics['fbeta_{}'.format(beta_option)] = fbeta_score(y_test, y_pred, beta=beta_option)
    return metrics
