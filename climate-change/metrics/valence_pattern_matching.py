# Calculates amount and ratio of positive and negative words using pattern matching with a dictionary

## PREREQUISITES
## python -m spacy download en

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time

CONTEXT_STEMMED_TWEETS_FILE = '../dataset/context_tweets_pattern_matching.csv'
VALENCE_WORDS_DICTIONARY = 'dictionaries/anew_val_polarity.xlsx'
OUTPUT_FILE = '../outputs/valence_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_STEMMED_TWEETS_FILE)
all_tweets = all_tweets.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
valence_dictionary = pd.read_excel(VALENCE_WORDS_DICTIONARY)
valence_dictionary = valence_dictionary[(valence_dictionary.Valence_standardized < -0.5) | (valence_dictionary.Valence_standardized > 1)]
valence_dictionary['Word'] = valence_dictionary['Word'].astype('str').str.lower()

negative_words = valence_dictionary[(valence_dictionary.Valence_standardized < -0.5)]['Word'].tolist()
positive_words = valence_dictionary[(valence_dictionary.Valence_standardized > 1)]['Word'].tolist()
negative_words.remove('nan')
negative_words = list(stemmer.stem(w) for w in negative_words)
positive_words = list(stemmer.stem(w) for w in positive_words)
negative_words = set(negative_words)
positive_words = set(positive_words)

all_tweets['negative_words'] = all_tweets['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(negative_words)))
all_tweets['positive_words'] = all_tweets['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(positive_words)))

all_tweets['negative_words_n'] = all_tweets['negative_words'].str.split().map(len)
all_tweets['negative_words_ratio'] = all_tweets['negative_words'].str.split().map(len) / all_tweets['stem_text'].str.split().map(len)

all_tweets['positive_words_n'] = all_tweets['positive_words'].str.split().map(len)
all_tweets['positive_words_ratio'] = all_tweets['positive_words'].str.split().map(len) / all_tweets['stem_text'].str.split().map(len)

all_tweets = all_tweets.drop(columns=['text', 'stem_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)
