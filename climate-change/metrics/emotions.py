# Uses distilroberta to get emotions for all tweets in the context. No preprocessing is done

import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

CONTEXT_TWEETS_FILE = '../dataset/context_tweets.csv'
EMOTIONS_OUTPUT_FILE = '../outputs/emotion_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_TWEETS_FILE)

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

all_tweets_emotions = []

for index, row in all_tweets.iterrows():
    tweet_emotions = classifier(row['text'])
    tweet_emotions = {emotion['label']: emotion['score'] for emotion in tweet_emotions[0]}
    tweet_emotions['id'] = row['id']

    all_tweets_emotions.append(tweet_emotions)

emotions_df = pd.DataFrame(all_tweets_emotions)
emotions_df.to_csv(EMOTIONS_OUTPUT_FILE, index=False)
