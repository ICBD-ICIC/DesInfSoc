# Calculates amount and ratio of moral words using pattern matching with a dictionary

## PREREQUISITES
## python -m spacy download en

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time

CONTEXT_STEMMED_TWEETS_FILE = '../dataset/context_tweets_pattern_matching.csv'
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
OUTPUT_FILE = '../outputs/mfd_{0}.csv'.format(time.time())

all_tweets = pd.read_csv(CONTEXT_STEMMED_TWEETS_FILE)
all_tweets = all_tweets.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
mfd_dictionary = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')
mfd_dictionary.Word = mfd_dictionary.Word.apply(lambda x: stemmer.stem(str(x).lower()))
mfd_dictionary = mfd_dictionary.drop_duplicates()

for category in ['vice', 'virtue']:
    words_mfd = set(mfd_dictionary[mfd_dictionary["Category"].str.contains(category)]['Word'].values)
    all_tweets['mfd_' + category] = all_tweets.stem_text.apply(lambda x: " ".join(set(str(x).split()).intersection(words_mfd)))
    all_tweets[category + '_n'] = all_tweets['mfd_' + category].str.split().map(len)
    all_tweets[category + '_ratio'] = all_tweets[category + '_n'].astype('int') / all_tweets['stem_text'].str.split().map(len)

all_tweets = all_tweets.drop(columns=['text', 'stem_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)
