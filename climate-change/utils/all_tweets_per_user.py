# Use this file to get all tweets concatenated for a given list of usernames USERS_LIST
# The output is a csv file located on OUTPUT_FILE path

import pandas as pd

USERS_LIST = '../dataset/users_min_10.csv'
PROFILES = '../dataset/profiles.csv'
REPLIES = '../dataset/replies.csv'
OUTPUT_FILE = '../dataset/users_min_10_all_tweets.csv'

usernames = pd.read_csv(USERS_LIST)
usernames = usernames[['user']].set_index('user')

profiles = pd.read_csv(PROFILES)
profiles = profiles[['username', 'text']]
replies = pd.read_csv(REPLIES)
replies = replies[['username', 'text']]

all_tweets = pd.concat([profiles, replies], ignore_index=True, sort=False).set_index('username')

all_tweets_per_user = all_tweets.groupby(['username'])['text'].apply(lambda x: '. '.join(x))

all_tweets_per_user_filtered = usernames.join(all_tweets_per_user)

all_tweets_per_user_filtered.to_csv(OUTPUT_FILE)

print(all_tweets_per_user_filtered)