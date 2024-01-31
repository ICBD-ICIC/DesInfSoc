import pandas as pd
import time
import sys
import csv
from datetime import timedelta
from ast import literal_eval
from discretize_tweets_metrics import *

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

K = 3  # number of intervals in context
H = 4  # amount of hours per interval
P = 12  # amount of hours of the prediction interval

interval_duration = timedelta(hours=H)
context_duration = timedelta(hours=H * K)
prediction_duration = timedelta(hours=P)

one_second = timedelta(seconds=1)

INTERVAL_SIZE = 3000
SPREAD = 60

interval_init = INTERVAL_SIZE * int(sys.argv[1])
interval_end = interval_init + INTERVAL_SIZE

USERS = pd.read_csv('dataset/user-metrics-with-friends.csv', converters={"friends": literal_eval})
USERS = USERS.iloc[interval_init:interval_end].set_index('user_id')

TWEETS = pd.read_csv('dataset/india-election-tweets-metrics.csv')  # Order by created_at ascending
TWEETS['created_at'] = pd.to_datetime(TWEETS['created_at'])
TWEETS = TWEETS.set_index('user_id')

OUTPUT_FILE = 'dataset/outputs/context_SPREAD60_K{0}_h{1}_interval_{2}-{3}_{4}.csv'.format(K, H, interval_init, interval_end, time.time())


def interval_tweets(friends_tweets, user_tweets):
    min_created_at = friends_tweets.iloc[0]['created_at']

    start_time = min_created_at.replace(hour=int(min_created_at.hour - (min_created_at.hour % (24 / H))),
                                        minute=0, second=0)
    end_time = start_time + context_duration
    max_created_at = max(friends_tweets.iloc[-1]['created_at'], end_time)
    prediction_end = end_time + prediction_duration

    intervals = []

    friends_tweets = friends_tweets.set_index('created_at')
    user_tweets = user_tweets.set_index('created_at')

    while end_time <= max_created_at:
        context_tweets = friends_tweets.loc[start_time:(end_time - one_second)]
        if len(context_tweets.index) != 0:
            prediction_tweets = user_tweets.loc[end_time:prediction_end]
            intervals.append((context_tweets, prediction_tweets))
        start_time += interval_duration
        end_time += interval_duration
        prediction_end += interval_duration
    return intervals


def context(tweets):
    return discretize_abusive(tweets) + \
           discretize_polarization(tweets) + \
           predominant_emotion(tweets) + \
           discretize_emotions(tweets) + \
           discretize_mfd(tweets) + \
           discretize_valence(tweets) + \
           predominant_sentiment(tweets) + \
           discretize_sentiments(tweets) + \
           (len(tweets),)


def ground_truth(tweets):
    return discretize_abusive(tweets) + \
           discretize_polarization(tweets) + \
           predominant_emotion(tweets) + \
           prediction_mfd(tweets) + \
           prediction_valence(tweets) + \
           predominant_sentiment(tweets) + \
           (len(tweets),)


with open(OUTPUT_FILE, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    for user_id, user in USERS.iterrows():
        print('Calculating for user: {0}'.format(user_id))
        time_start = time.time()

        user_context = user['big_five'], user['symanto_psychographics']

        friends_tweets = TWEETS[TWEETS.index.isin(user['friends'])]
        if len(friends_tweets) != 0:
            user_tweets = TWEETS.loc[user_id]
            intervals = interval_tweets(friends_tweets, user_tweets)

            # Spread
            empty_ground_truth = sum(1 for interval in intervals if interval[1].empty)
            percentage_empty = (empty_ground_truth / len(intervals)) * 100
            percentage_non_empty = 100 - percentage_empty
            if abs(percentage_empty - percentage_non_empty) > SPREAD:
                continue

            for interval in intervals:
                context_values = context(interval[0])
                ground_truth_values = ground_truth(interval[1])
                context_prediction = user_context + context_values + ground_truth_values
                csv_writer.writerow(context_prediction)
        print('Finish calculating user: {0}. Total seconds: {1}'.format(user_id, time.time()-time_start))
print('FINISHED')
