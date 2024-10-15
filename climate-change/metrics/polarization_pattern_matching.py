# Calculates amount and ratio of polar words using pattern matching with a dictionary

## PREREQUISITES
## python -m spacy download en

import pandas as pd

from nltk.stem.snowball import SnowballStemmer
import time

CONTEXT_STEMMED_TWEETS_FILE = '../dataset/context_tweets_pattern_matching.csv'
POLARIZATION_WORDS_DICTIONARY = 'dictionaries/lang_online_polarization_dict.csv'
OUTPUT_FILE = '../outputs/polar_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_STEMMED_TWEETS_FILE)
all_tweets = all_tweets.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
polarization_dictionary = pd.read_csv(POLARIZATION_WORDS_DICTIONARY)
polarization_words = polarization_dictionary.word.values
polarization_words = list(stemmer.stem(w.lower()) for w in polarization_words)
polarization_words = set(polarization_words)

all_tweets['polar_words'] = all_tweets['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(polarization_words)))
all_tweets['polar_words_n'] = all_tweets['polar_words'].str.split().map(len)
all_tweets['polar_words_ratio'] = all_tweets['polar_words_n'].astype('int') / all_tweets['stem_text'].str.split().map(len)

all_tweets = all_tweets.drop(columns=['text', 'stem_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)

