import sys
import pandas as pd
import time

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, auc,
                             precision_recall_curve)

KFOLD = 3
TEST_SIZE = 0.1
CONTEXT_MAP = {
    0: 'user_big_five',
    1: 'user_symanto_psychographics',
    2: 'abusive_amount_interval',
    3: 'abusive_ratio_interval',
    4: 'abusive_words_present',
    5: 'polarization_amount_interval',
    6: 'polarization_ratio_interval',
    7: 'polarization_words_present',
    8: 'predominant_emotion',
    9: 'emotion_neutral_interval',
    10: 'emotion_anger_interval',
    11: 'emotion_disgust_interval',
    12: 'emotion_fear_interval',
    13: 'emotion_joy_interval',
    14: 'emotion_sadness_interval',
    15: 'emotion_surprise_interval',
    16: 'mfd_virtue_amount',
    17: 'mfd_virtue_ratio',
    18: 'mdf_virtue_words_present',
    19: 'mfd_vice_amount',
    20: 'mfd_vice_ratio',
    21: 'mfd_vice_words_present',
    22: 'valence_positive_amount',
    23: 'valence_positive_ratio',
    24: 'positive_words_present',
    25: 'valence_negative_amount',
    26: 'valence_negative_ratio',
    27: 'negative_words_present',
    28: 'predominant_sentiment',
    29: 'sentiment_neutral_interval',
    30: 'sentiment_positive_interval',
    31: 'sentiment_negative_interval',
    32: 'context_tweets_amount'
}
PREDICTION_MAP = {
    33: 'abusive_amount_interval',
    34: 'abusive_ratio_interval',
    35: 'abusive_words_present',
    36: 'polarization_amount_interval',
    37: 'polarization_ratio_interval',
    38: 'polarization_words_present',
    39: 'predominant_emotion',
    40: 'mfd_ratio',
    41: 'mfd_amount',
    42: 'has_virtue',
    43: 'has_vice',
    44: 'valence_ratio',
    45: 'valence_amount',
    46: 'has_positive',
    47: 'has_negative',
    48: 'predominant_sentiment',
    49: 'prediction_tweets_amount'
}
# Defines which features to check to know if the feature is present
PREDICTION_EXISTENCE = {
    33: ['35'],
    34: ['35'],
    36: ['38'],
    37: ['38'],
    40: ['42', '43'],
    41: ['42', '43'],
    44: ['46', '47'],
    45: ['46', '47']
}

BETA_OPTIONS = [0.5, 2, 3, 4]

context_columns = list(range(0, 32))

prediction = int(sys.argv[1])
dataset_name = sys.argv[2]


def precision_recall_auc(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


def prediction_present(dataset):
    if prediction in PREDICTION_EXISTENCE.keys():
        features = PREDICTION_EXISTENCE[prediction]
        if len(features) == 1:
            return dataset[dataset[features[0]] == 1]
        else:
            return dataset[(dataset[features[0]] == 1) | (dataset[features[1]] == 1)]
    else:
        return dataset


def get_dataset():
    dataset = pd.read_csv('dataset/{}.csv'.format(dataset_name))
    dataset = prediction_present(dataset)
    X = dataset.iloc[:, context_columns]
    y = dataset.iloc[:, prediction]
    return X, y


def get_train_test_split():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # print('TOTAL Amount per class, original:')
    # print(y.value_counts())
    # print('TRAIN Amount per class, original:')
    # print(y_train.value_counts())
    # X_train, y_train = RandomUnderSampler(random_state=42, sampling_strategy=1).fit_resample(X_train, y_train)
    # print('TRAIN Amount per class, after random under sampler:')
    # print(y_train.value_counts())

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
