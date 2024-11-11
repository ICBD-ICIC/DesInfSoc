# Creates the dataset as needed for the LLM classifiers

from ast import literal_eval
import pandas as pd
import time
from discretization.discretize_tweets_metrics import *

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

experiment_type = 'distance'

USERS_TWEETS = pd.read_csv('dataset/for_llm_context/users_min_10_all_tweets.csv')
TWEETS_IDS = pd.read_csv('dataset/for_llm_context/input_and_ground_truth.csv',
                         converters={"previous_posts_ids": literal_eval})
CONTEXT_TWEETS = pd.read_csv('dataset/for_llm_context/context_tweets.csv')
TWEETS_FEATURES = pd.read_csv('dataset/for_context/tweets_features_{}.csv'.format(experiment_type)) # for predictions

OUTPUT_FILE = 'outputs/CONTEXT_LLM_{}_{}.csv'.format(experiment_type, time.time())

discretizer = TweetsMetricsDiscretizer(experiment_type)

#Dessigned for multiple tweets but in our case we have only 1
def ground_truth(tweets):
    all_ground_truths = { **discretizer.discretize_abusive(tweets),
                         **discretizer.discretize_polarization(tweets),
                         **discretizer.predominant_emotion(tweets),
                         **discretizer.prediction_mfd(tweets),
                         **discretizer.prediction_valence(tweets),
                         **discretizer.predominant_sentiment(tweets) }
    return { str(key) + '_gt': val for key, val in all_ground_truths.items()}

context_rows = []

for index, row in TWEETS_IDS.iterrows():
    user_tweets = USERS_TWEETS[USERS_TWEETS['user'] == row['username']]['text'].iloc[0]
    prediction_tweets = TWEETS_FEATURES[TWEETS_FEATURES['id'] == row['user_reply_id']]
    context_tweets = ". ".join(CONTEXT_TWEETS[CONTEXT_TWEETS['id'].isin(row['previous_posts_ids'])]['text'].tolist())
    context_rows.append({'text': user_tweets + context_tweets,
                          **ground_truth(prediction_tweets)})
context_df = pd.DataFrame(context_rows)
context_df.to_csv(OUTPUT_FILE, index=False)