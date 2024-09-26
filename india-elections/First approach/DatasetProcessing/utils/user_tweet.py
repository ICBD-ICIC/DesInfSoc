# Use this file to extract user id and tweet from the india-election-tweets.json dataset
# The output is a csv file located on USERS_TWEETS_FILE path

import json
import csv

USERS_TWEETS_FILE = 'outputs/user_tweets.csv'
FAILED_LINES_FILE = 'dataset/india-election-tweets-failed.json'

with open('dataset/india-election-tweets.json', 'r', encoding="utf8") as tweets_file:
    user_tweets = open(USERS_TWEETS_FILE, 'w', encoding="utf8")
    csv_writer = csv.writer(user_tweets)
    csv_writer.writerow(['creator_id', 'tweet_content'])
    failed_lines = open(FAILED_LINES_FILE, 'w', encoding="utf8")
    try:
        for line in tweets_file:
            json_line = json.loads(line)
            # Remove the following if statement if you want the RTs to be included
            if 'retweeted_status' in json_line:
                continue
            try:
                creator_id = json_line["user"]["id"]["$numberLong"]
            except TypeError:
                creator_id = json_line["user"]["id"]
            tweet_content = json_line["text"]
            csv_writer.writerow([creator_id, tweet_content])
    except Exception as e:
        print(e)
        print(line)
        json.dump(line, failed_lines)
    user_tweets.close()
    failed_lines.close()
