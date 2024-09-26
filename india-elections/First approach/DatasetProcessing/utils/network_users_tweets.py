# Use this file to get all available text from USERS_CONCATENATED_TWEETS_FILE for the users present on the network
# The output is a csv file located on NETWORK_USERS_TWEETS_FILE path with all text available for the users, and a
# list of users without tweets located on USERS_WITHOUT_TWEETS_FILE path

import pandas as pd

USERS_CONCATENATED_TWEETS_FILE = 'dataset/all-languages/all_tweets_per_user.csv'

NETWORK_USERS_TWEETS_FILE = 'outputs/network_users_tweets.csv'
USERS_WITHOUT_TWEETS_FILE = 'outputs/users_without_tweets.csv'

network = pd.read_json('dataset/india_network.json', lines=True)
users_tweets = pd.read_csv(USERS_CONCATENATED_TWEETS_FILE)

# Join data from network users and user tweets
all_network_users = network.set_index('id').join(users_tweets.set_index('creator_id'), how='left').reset_index()

# Get users that have tweet content
users_with_tweets = all_network_users[all_network_users['tweet_content'].notnull()][['id', 'tweet_content']]
# print(users_with_tweets) [143614 rows x 2 columns]
users_with_tweets.to_csv(NETWORK_USERS_TWEETS_FILE, index=False)

# Get users that do not have tweets content
users_without_tweets = all_network_users[all_network_users['tweet_content'].isnull()]['id']
users_without_tweets.to_csv(USERS_WITHOUT_TWEETS_FILE, index=False)
