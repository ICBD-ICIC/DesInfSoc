# Use this file to get number of tweets for all users and only network users, in decrement order
# The output are 2 csv files located on USERS_ACTIVITY_FILE and NETWORK_USERS_ACTIVITY path

import pandas as pd

USERS_TWEETS_FILE = 'dataset/all-languages/user_tweets.csv'
ALL_USERS_ACTIVITY_FILE = 'outputs/all_users_by_activity.csv'
NETWORK_USERS_ACTIVITY = 'outputs/network_users_by_activity.csv'

#####################
##### ALL USERS #####
#####################

users_tweets = pd.read_csv(USERS_TWEETS_FILE)

# get all messages by user id
users_tweets = users_tweets.groupby(['creator_id']).count()

# Sort descending
users_tweets = users_tweets.sort_values(by='tweet_content', ascending=False)

# save as csv
users_tweets.to_csv(ALL_USERS_ACTIVITY_FILE)

#####################
### NETWORK USERS ###
#####################

network = pd.read_json('dataset/india_network.json', lines=True)

# filter users present on the network
all_network_users = network.set_index('id').join(users_tweets, how='left').reset_index()
# remove NaNs (users without tweets)
all_network_users = all_network_users[all_network_users['tweet_content'].notnull()][['id', 'tweet_content']].astype(int)
# rename column
all_network_users = all_network_users.rename(columns={"tweet_content": 'tweet_amount'})
# sort data
all_network_users = all_network_users.sort_values(by='tweet_amount', ascending=False)
# save results
all_network_users.to_csv(NETWORK_USERS_ACTIVITY, index=False)
