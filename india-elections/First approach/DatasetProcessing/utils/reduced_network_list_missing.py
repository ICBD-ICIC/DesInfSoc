import pandas as pd
import json
import csv

SELECTED_USERS = '../dataset/network_active_users.csv'
FILTERED_TWEETS = '../dataset/india-election-tweets-formatted-filtered-clean.csv'
TWEETS_MISSING = '../outputs/india-election-tweets-formatted-missing.csv'

users = pd.read_csv(SELECTED_USERS).set_index('id').index.tolist()
filtered_tweets = pd.read_csv(FILTERED_TWEETS)

missing_users = list(set(users) - set(list(filtered_tweets['user_id'])))

formatted_dataset = open(TWEETS_MISSING, 'w', encoding="utf8")
csv_writer = csv.writer(formatted_dataset)
csv_writer.writerow(['id',
                     'created_at',
                     'text',
                     'username',
                     'user_id'])

with open('../dataset/india-election-tweets.json', 'r', encoding="utf8") as tweets_file:
    for line in tweets_file:
        json_line = json.loads(line)
        if int(json_line["user"]["id_str"]) in missing_users:
            csv_writer.writerow([json_line["id_str"],
                                 json_line["created_at"],
                                 json_line["text"],
                                 json_line["user"]["screen_name"],
                                 json_line["user"]["id_str"]])

formatted_dataset.close()
