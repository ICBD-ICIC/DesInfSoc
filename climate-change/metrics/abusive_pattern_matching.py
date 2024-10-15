# Calculates amount and ratio of abusive words using pattern matching with a dictionary

## PREREQUISITES
## python -m spacy download en

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time

CONTEXT_STEMMED_TWEETS_FILE = '../dataset/context_tweets_pattern_matching.csv'
ABUSIVE_WORDS_DICTIONARY = 'dictionaries/abuseLexicon.xlsx'
OUTPUT_FILE = '../outputs/abusive_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_STEMMED_TWEETS_FILE)
all_tweets = all_tweets.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')

abusive_words_dictionary = pd.read_excel(ABUSIVE_WORDS_DICTIONARY)
abusive_words = abusive_words_dictionary.word.str.lower().values
abusive_words = list(stemmer.stem(w) for w in abusive_words)
abusive_words = set(abusive_words)

all_tweets['abusive_words'] = all_tweets['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(abusive_words)))
all_tweets['abusive_words_n'] = all_tweets['abusive_words'].str.split().map(len)
all_tweets['abusive_words_ratio'] = all_tweets['abusive_words_n'].astype('int') / all_tweets['stem_text'].str.split().map(len)

num_cols = all_tweets.describe().columns
all_tweets[num_cols] = all_tweets[num_cols].fillna(0).round(4)

output_file = '../outputs/itrust/abusive/itrust_metrics_abusive_{0}.csv'.format(time.time())
all_tweets = all_tweets.drop(columns=['text', 'stem_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)
