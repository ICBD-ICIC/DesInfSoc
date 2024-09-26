# Use this file to get all tweets concatenated for a given list of usernames USERS_LIST
# The output is a csv file located on OUTPUT_FILE path

import pandas as pd

USERS_LIST = '../dataset/users_min_10.csv'
PROFILES = '../dataset/profiles.xlsx'
REPLIES = '../dataset/replies.xlsx'
OUTPUT_FILE = '../dataset/users_min_10_all_tweets.csv'

profiles = pd.read_excel(PROFILES)['username', 'text']
# replies = pd.read_excel(REPLIES)
#
# # # drop all RTs
# # df = df[df["tweet_content"].str.startswith("RT @") == False]
#
# # get all messages by user id
# df = df.groupby(['creator_id'])['tweet_content'].apply(lambda x: '. '.join(x)).reset_index()
#
# # save as csv
# df.to_csv(USERS_CONCATENATED_TWEETS_FILE, index=False)

print(profiles)