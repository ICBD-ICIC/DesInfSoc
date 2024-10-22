# For each user, gets the tweets ids of the conversation before its reply

import pandas as pd
pd.set_option("max_colwidth", 500)

CONTEXT_TWEETS = 'dataset/context_tweets_pattern_matching.csv'
USERS = 'dataset/users_min_10.csv'
REPLIES = 'dataset/replies.csv'
INFLUENCERS = 'dataset/influencers.csv'
OUTPUT_FILE = 'dataset/input_and_ground_truth.csv'

usernames = pd.read_csv(USERS)['user']
replies = pd.read_csv(REPLIES)
influencers = pd.read_csv(INFLUENCERS)

input_and_ground_truth = []

for username in usernames:
    user_replies = replies[replies['username'] == username]
    for index, reply in user_replies.iterrows():
        conversation_id = reply['conversation_id']
        original_tweet = influencers[influencers['id'] == conversation_id]
        all_tweets_conversation = replies[replies['conversation_id'] == conversation_id]
        previous_posts_ids = original_tweet['id'].to_list()
        previous_posts_ids += all_tweets_conversation[all_tweets_conversation['created_at'] < reply['created_at']]['id'].to_list()
        if len(previous_posts_ids) > 0:
            input_and_ground_truth.append({'username': username,
                                           'user_reply_id': reply['id'],
                                           'previous_posts_ids': previous_posts_ids})
input_and_ground_truth_df = pd.DataFrame(input_and_ground_truth)
input_and_ground_truth_df.to_csv(OUTPUT_FILE, index=False)
print(len(input_and_ground_truth_df.index))