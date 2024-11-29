# Use this file to get all tweets used as part of the context
# The output is a csv file located on OUTPUT_FILE path

import pandas as pd

INFLUENCERS = '../dataset/influencers.csv'
REPLIES = '../dataset/replies.csv'
OUTPUT_FILE = '../dataset/context_tweets_llm.csv'

influencers = pd.read_csv(INFLUENCERS)
influencers = influencers[['id', 'text', 'created_at', 'username']]
replies = pd.read_csv(REPLIES)
replies = replies[['id', 'text', 'created_at', 'username']]

all_tweets = pd.concat([influencers, replies], ignore_index=True, sort=False).set_index('id')

all_tweets.to_csv(OUTPUT_FILE)

print(all_tweets)