# Creates the dataset as needed for the ML classifiers

from ast import literal_eval
import pandas as pd
import time
from discretization.discretize_tweets_metrics import *

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

experiment_type = 'pattern_matching'

USERS = pd.read_csv('dataset/for_context/user_features.csv')
TWEETS_IDS = pd.read_csv('dataset/for_context/input_and_ground_truth.csv',
                         converters={"previous_posts_ids": literal_eval})
TWEETS_FEATURES = pd.read_csv('dataset/for_context/tweets_features.csv')

OUTPUT_FILE = 'outputs/context_{}_{}.csv'.format(experiment_type, time.time())

def context(tweets):
    return { **discretize_abusive(tweets),
             **discretize_polarization(tweets),
             **predominant_emotion(tweets),
             **discretize_emotions(tweets),
             **discretize_mfd(tweets),
             **discretize_valence(tweets),
             **predominant_sentiment(tweets),
             **discretize_sentiments(tweets),
             'context_amount': len(tweets) }

#Dessigned for multiple tweets but in our case we have only 1
def ground_truth(tweets):
    all_ground_truths = { **discretize_abusive(tweets),
                         **discretize_polarization(tweets),
                         **predominant_emotion(tweets),
                         **prediction_mfd(tweets),
                         **prediction_valence(tweets),
                         **predominant_sentiment(tweets) }
    return { str(key) + '_gt': val for key, val in all_ground_truths.items()}

context_rows = []

for index, row in TWEETS_IDS.iterrows():
    user_features = USERS[USERS['username'] == row['username']]
    prediction_tweets = TWEETS_FEATURES[TWEETS_FEATURES['id'] == row['user_reply_id']]
    ground_truth_tweets = TWEETS_FEATURES[TWEETS_FEATURES['id'].isin(row['previous_posts_ids'])]
    context_rows.append({ 'big_five': int(user_features['big_five']),
                          'psychographics': int(user_features['psychographics']),
                          **context(ground_truth_tweets),
                          **ground_truth(prediction_tweets)})
context_df = pd.DataFrame(context_rows)
context_df.to_csv(OUTPUT_FILE, index=False)