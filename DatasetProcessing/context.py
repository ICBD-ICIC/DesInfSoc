import pandas as pd
import time
from datetime import timedelta
# from discretize_personality import *
from discretize_tweets_metrics import *

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

K = 3  # number of intervals in context
H = 4  # amount of hours per interval

ALL_USERS = pd.read_csv('dataset/network_active_users.csv').set_index('id')[0:3]
NETWORK = pd.read_json('dataset/india_network.json', lines=True).set_index('id')
TWEETS = pd.read_csv('dataset/india-election-tweets-formatted-filtered-clean.csv').set_index('id')
TWEETS['created_at'] = pd.to_datetime(TWEETS['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
# TWEETS = pd.read_csv('outputs/text.csv').set_index('id')
# TWEETS['created_at'] = pd.to_datetime(TWEETS['created_at'])
TWEETS = TWEETS[['user_id', 'created_at']]
TWEETS = TWEETS.sort_values(by=['created_at'])
OUTPUT_FILE = 'outputs/context_K{0}_h{1}_{2}'.format(K, H, time.time())


# def is_duplicated(intervals, interval):
#     return len(intervals) != 0 and intervals[-1] == interval


def interval_tweets(friends, user_id):
    friends_tweets = TWEETS[TWEETS['user_id'].isin(friends)]
    user_tweets = TWEETS[TWEETS['user_id'] == user_id]

    user_tweets.to_csv('outputs/test2.csv')

    min_created_at = friends_tweets.iloc[0]['created_at']
    max_created_at = friends_tweets.iloc[-1]['created_at']

    interval_duration = timedelta(hours=H)
    total_interval_duration = timedelta(hours=H*K)

    start_time = min_created_at.replace(hour=int(min_created_at.hour - (min_created_at.hour % (24/H))),
                                        minute=0, second=0)
    end_time = start_time + total_interval_duration

    intervals = []

    while end_time < max_created_at:
        context_tweets = friends_tweets.loc[(friends_tweets['created_at'] >= start_time) &
                                            (friends_tweets['created_at'] < end_time)]
        start_time += interval_duration
        end_time += interval_duration
        context_tweets_ids = context_tweets.index.tolist()
        if len(context_tweets_ids) != 0:
            prediction = user_tweets.loc[(user_tweets['created_at'] >= end_time) &
                                         (user_tweets['created_at'] < (end_time + interval_duration))]
            prediction_ids = prediction.index.tolist()
            intervals.append({'context_tweets_ids': context_tweets_ids, 'prediction_ids': prediction_ids})
    return intervals


# def context(tweets_ids):
#     return {
#         **discretize_abusive(tweets_ids),
#         **discretize_polarization(tweets_ids),
#         **predominant_emotion(tweets_ids),
#         **discretize_emotions(tweets_ids),
#         **discretize_mfd(tweets_ids),
#         **discretize_valence(tweets_ids),
#         **predominant_sentiment(tweets_ids),
#         **discretize_sentiments(tweets_ids),
#         'tweets_amount': len(tweets_ids)
#     }

def ground_truth(tweets_ids):
    prediction = {
        **discretize_abusive(tweets_ids),
        **discretize_polarization(tweets_ids),
        **prediction_mfd(tweets_ids),
        **prediction_valence(tweets_ids),
        **predominant_sentiment(tweets_ids),
        'tweets_amount': len(tweets_ids)
        # falta emotions
    }
    ground_truth = {f"ground_truth_{key}": val for key, val in prediction.items()}
    return ground_truth


for user_id in ALL_USERS.index:
    user_context = {
        # **discretize_big_five(user_id),
        # **discretize_psychographics(user_id)
    }

    user_network = NETWORK.loc[user_id]
    if user_network['friends_count'] > 0:
        intervals = interval_tweets(user_network['friends'], user_id)
        # context = list(map(lambda tweets_ids: context(tweets_ids['context_tweets_ids']), intervals))
        ground_truth = list(map(lambda tweets_ids: ground_truth(tweets_ids['prediction_ids']), intervals))
        print(user_context)
        # print(tweets_context)
        print(ground_truth)
