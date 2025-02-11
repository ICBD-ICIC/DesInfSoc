import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
discretization_path = os.path.join(current_dir, '..', '..', 'discretization')
sys.path.append(discretization_path)

from discretize_tweets_metrics import *

DATASET = 'dataset/CONTEXT_LLM_pattern_matching_test.csv'

HIGH_LABEL = 'High'
LOW_LABEL = 'Low'
EXPERIMENT_TYPE = 'pattern_matching'

features = ['abusive', 'polarization', 'emotion', 'valence', 'sentiment', 'mfd']

discretizer = TweetsMetricsDiscretizer(EXPERIMENT_TYPE)

def get_feature_type(feature):
    if feature in ['sentiment', 'emotion']:
        return 'categorical'
    if feature in ['abusive', 'polarization']:
        return 'interval'
    if feature in ['valence', 'mfd']:
        return 'interval_comparison'


def ground_truth(data, feature):
    feature_type = get_feature_type(feature)
    if feature_type == 'interval':
        if (data[f'{feature}_amount_interval_gt'] > 1) or (data[f'{feature}_ratio_interval_gt'] > 1):
            return HIGH_LABEL
        else:
            return LOW_LABEL
    if feature_type == 'categorical':
        if feature == 'sentiment':
            return discretizer.sentiment_categories[data['predominant_sentiment_gt']].replace('sentiment-', '').capitalize()
        elif feature == 'emotion':
            return discretizer.emotions_categories[data['predominant_emotion_gt']].capitalize()
    if feature_type == 'interval_comparison':
        if feature == 'mfd':
            category1 = 'Virtue'
            category2 = 'Vice'
        elif feature == 'valence':
            category1 = 'Positive'
            category2 = 'Negative'
        if data[f'{feature}_amount_gt'] == 0:
            return "Equal"
        elif data[f'{feature}_amount_gt'] == 1:
            return category1
        elif data[f'{feature}_amount_gt'] == 2:
            return category2


df = pd.read_csv(DATASET)

gt_columns = [col for col in df.columns if "_gt" in col]

print(DATASET)

for col in features:
    df[col] = df.apply(lambda x: ground_truth(x, col), axis=1)
    print(df[col].value_counts())
