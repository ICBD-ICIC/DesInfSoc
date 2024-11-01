# Calculates amount and ratio of MFD virtue words using distance words embeddings with a dictionary

# PREREQUISITES
# Download crawl-300d-2M-subword.zip from https://fasttext.cc/docs/en/english-vectors.html and save te .bin into MODEL_PATH

import pandas as pd
import time
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

already_found_words = []
already_checked = []

# Dictionary is a list of embeddings with the dictionary words for the linguistic feature being calculated
def mfd_virtue_words(text, dictionary):
    mfd_virtue_or_similar = []
    for word in text:
        found = False
        if word in already_found_words:
            mfd_virtue_or_similar.append(word)
            found = True
        elif not word in already_checked:
            word_embedding = ft.get_word_vector(word).reshape(1, -1)
            for dictionary_word in dictionary:
                if cosine_similarity(word_embedding, dictionary_word) >= 0.6:
                    mfd_virtue_or_similar.append(word)
                    already_found_words.append(word)
                    found = True
                    break
        if not found:
            already_checked.append(word)
    return " ".join(mfd_virtue_or_similar)

# Local paths
MODEL_PATH = '../../../crawl-300d-2M-subword.bin'
CONTEXT_CLEANED_TWEETS_FILE = '../dataset/context_tweets_distance.csv'
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
OUTPUT_FILE = '../outputs/mfd_virtue_{0}.csv'.format(time.time())

# Paths in the kluster
# MODEL_PATH = 'crawl-300d-2M-subword.bin'
# CONTEXT_CLEANED_TWEETS_FILE = 'dataset/context_tweets_distance.csv'
# MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
# OUTPUT_FILE = 'outputs/mfd_virtue_{0}.csv'.format(time.time())

ft = fasttext.load_model(MODEL_PATH)

all_tweets = pd.read_csv(CONTEXT_CLEANED_TWEETS_FILE, converters={"clean_text": literal_eval})[0:10]

start = time.time()
mfd_dictionary = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')
mfd_dictionary.Word = mfd_dictionary.Word.apply(lambda x: str(x).lower())
virtue_words = set(mfd_dictionary[mfd_dictionary["Category"].str.contains('virtue')]['Word'].values)
virtue_embeddings = [ft.get_word_vector(word).reshape(1, -1) for word in virtue_words]

all_tweets['mfd_virtue'] = all_tweets['clean_text'].apply(lambda text: mfd_virtue_words(text, virtue_embeddings))
all_tweets['virtue_n'] = all_tweets['mfd_virtue'].str.split().map(len)
all_tweets['virtue_ratio'] = all_tweets['virtue_n'].astype('int') / all_tweets['clean_text'].map(len)

num_cols = all_tweets.describe().columns
all_tweets[num_cols] = all_tweets[num_cols].fillna(0).round(4)

all_tweets = all_tweets.drop(columns=['clean_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)

print('FINISHED after {} seconds'.format(time.time()-start))
