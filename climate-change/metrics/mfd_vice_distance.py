# Calculates amount and ratio of MFD vice words using distance words embeddings with a dictionary

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
def mfd_vice_words(text, dictionary):
    mfd_vice_or_similar = []
    for word in text:
        found = False
        if word in already_found_words:
            mfd_vice_or_similar.append(word)
            found = True
        elif not word in already_checked:
            word_embedding = ft.get_word_vector(word).reshape(1, -1)
            for dictionary_word in dictionary:
                if cosine_similarity(word_embedding, dictionary_word) >= 0.6:
                    mfd_vice_or_similar.append(word)
                    already_found_words.append(word)
                    found = True
                    break
        if not found:
            already_checked.append(word)
    return " ".join(mfd_vice_or_similar)

# Local paths
# MODEL_PATH = '../../../crawl-300d-2M-subword.bin'
# CONTEXT_CLEANED_TWEETS_FILE = '../dataset/context_tweets_distance.csv'
# MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
# OUTPUT_FILE = '../outputs/mfd_vice_{0}.csv'.format(time.time())

# Paths in the kluster
MODEL_PATH = 'crawl-300d-2M-subword.bin'
CONTEXT_CLEANED_TWEETS_FILE = 'dataset/context_tweets_distance.csv'
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
OUTPUT_FILE = 'outputs/mfd_vice_{0}.csv'.format(time.time())

ft = fasttext.load_model(MODEL_PATH)

all_tweets = pd.read_csv(CONTEXT_CLEANED_TWEETS_FILE, converters={"clean_text": literal_eval})

start = time.time()
mfd_dictionary = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')
mfd_dictionary.Word = mfd_dictionary.Word.apply(lambda x: str(x).lower())
vice_words = set(mfd_dictionary[mfd_dictionary["Category"].str.contains('vice')]['Word'].values)
vice_embeddings = [ft.get_word_vector(word).reshape(1, -1) for word in vice_words]

all_tweets['mfd_vice'] = all_tweets['clean_text'].apply(lambda text: mfd_vice_words(text, vice_embeddings))
all_tweets['vice_n'] = all_tweets['mfd_vice'].str.split().map(len)
all_tweets['vice_ratio'] = all_tweets['vice_n'].astype('int') / all_tweets['clean_text'].map(len)

num_cols = all_tweets.describe().columns
all_tweets[num_cols] = all_tweets[num_cols].fillna(0).round(4)

all_tweets = all_tweets.drop(columns=['clean_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)

print('FINISHED after {} seconds'.format(time.time()-start))
