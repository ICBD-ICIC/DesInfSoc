# Use this file to extract only english tweets from india-election-tweets.json dataset
# The output is a json file located on ONLY_ENGLISH_TWEETS_FILE path

import json

ONLY_ENGLISH_TWEETS_FILE = 'dataset/india-election-tweets-reduced-en.json'

with open('dataset/india-election-tweets-reduced.json', 'r', encoding="utf8") as tweets_file:
    english_tweets = open(ONLY_ENGLISH_TWEETS_FILE, 'w', encoding="utf8")
    for line in tweets_file:
        json_line = json.loads(line)
        if json_line["user"]["lang"] == 'en':
            print(json_line["user"]["lang"])
            english_tweets.write(line)
    english_tweets.close()
