import pandas as pd
import itertools

SELECTED_USERS = '../dataset/all_users_by_activity.csv'
FULL_NETWORK = '../dataset/india_network.json'
ALL_TWEETS = '../dataset/india-election-tweets-formatted.csv'
TWEETS_FILTERED = '../outputs/india-election-tweets-formatted-filtered.csv'

network = pd.read_json(FULL_NETWORK, lines=True).set_index('id')
selected_users = pd.read_csv(SELECTED_USERS)
selected_users = selected_users.where(selected_users['tweet_content'] >= 20).dropna().set_index('creator_id')

result = selected_users.join(network, how='inner').dropna()

friends = result['friends'].tolist()  # list of lists
friends = list(itertools.chain(*friends))  # flatten list
friends = list(dict.fromkeys(friends))  # remove duplicates

all_tweets = pd.read_csv(ALL_TWEETS)
filtered_tweets = all_tweets.where(all_tweets['user_id'].isin(friends)).dropna()
filtered_tweets = filtered_tweets.astype({'id': "Int64", 'user_id': "Int64"})
filtered_tweets.to_csv(TWEETS_FILTERED, index=False)

print(filtered_tweets)
