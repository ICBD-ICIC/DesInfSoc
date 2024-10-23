# Use this file to clean text to then calculate distance of word embeddings
import string
from string import punctuation

import pandas as pd
import time
from nltk.tokenize import TweetTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

CONTEXT_TWEETS_FILE = '../dataset/context_tweets.csv'
OUTPUT_FILE = '../outputs/context_tweets_distance_{0}.csv'.format(time.time())

df = pd.read_csv(CONTEXT_TWEETS_FILE)

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

stop_words = set(stopwords.words("english"))

# Tokenize
df['clean_text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
# Remove stop words
df['clean_text'] = df['clean_text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
# Remove punctuation tokens
punctuation_tokens = string.punctuation+'–'+'…'+'…'+'...'+'—'
df['clean_text'] = df['clean_text'].apply(lambda tokens: [word for word in tokens if word not in punctuation_tokens])
# Remove urls
df['clean_text'] = df['clean_text'].apply(lambda tokens: [token for token in tokens if not token.startswith("https:")])
df = df.drop(columns=['text'])
df.to_csv(OUTPUT_FILE, index=False)












