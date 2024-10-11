# Uses pysentimiento to get sentiments for all tweets in the context. No preprocessing is done

import pandas as pd
import time
from pysentimiento import create_analyzer
import transformers

CONTEXT_TWEETS_FILE = '../dataset/context_tweets.csv'
SENTIMENTS_OUTPUT_FILE = '../outputs/sentiment_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_TWEETS_FILE)

transformers.logging.set_verbosity(transformers.logging.ERROR)
analyzer = create_analyzer(task="sentiment", lang="en")

all_tweets_sentiments = []

for index, row in all_tweets.iterrows():
    tweet_sentiments = analyzer.predict(row['text'])
    all_tweets_sentiments.append({'id': row['id'],
                                  'sentiment-positive': tweet_sentiments.probas['POS'],
                                  'sentiment-negative': tweet_sentiments.probas['NEG'],
                                  'sentiment-neutral': tweet_sentiments.probas['NEU']})

sentiments_df = pd.DataFrame(all_tweets_sentiments)
sentiments_df.to_csv(SENTIMENTS_OUTPUT_FILE, index=False)
