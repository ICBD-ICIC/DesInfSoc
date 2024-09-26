# Use this file to get all tweets concatenated for all users
# The output is a csv file located on USERS_CONCATENATED_TWEETS_FILE path

import pandas as pd

USERS_TWEETS_FILE = 'dataset/all-languages/user_tweets.csv'
USERS_CONCATENATED_TWEETS_FILE = 'outputs/all_tweets_per_user.csv'

df = pd.read_csv(USERS_TWEETS_FILE)

# # drop all RTs
# df = df[df["tweet_content"].str.startswith("RT @") == False]

# get all messages by user id
df = df.groupby(['creator_id'])['tweet_content'].apply(lambda x: '. '.join(x)).reset_index()

# save as csv
df.to_csv(USERS_CONCATENATED_TWEETS_FILE, index=False)
