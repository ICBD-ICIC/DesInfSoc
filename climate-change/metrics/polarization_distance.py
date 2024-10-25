# Calculates amount and ratio of polar words using distance words embeddings with a dictionary

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
def polar_words(text, dictionary):
    polar_or_similar = []
    for word in text:
        found = False
        if word in already_found_words:
            polar_or_similar.append(word)
            found = True
        elif not word in already_checked:
            word_embedding = ft.get_word_vector(word).reshape(1, -1)
            for dictionary_word in dictionary:
                if cosine_similarity(word_embedding, dictionary_word) >= 0.6:
                    polar_or_similar.append(word)
                    already_found_words.append(word)
                    found = True
                    break
        if not found:
            already_checked.append(word)
    return " ".join(polar_or_similar)

MODEL_PATH = '../../../crawl-300d-2M-subword.bin'
CONTEXT_CLEANED_TWEETS_FILE = '../dataset/context_tweets_distance.csv'
POLARIZATION_WORDS_DICTIONARY = 'dictionaries/lang_online_polarization_dict.csv'
OUTPUT_FILE = '../outputs/polar_{0}.csv'.format(time.time())

# Paths in the kluster
# MODEL_PATH = 'crawl-300d-2M-subword.bin'
# CONTEXT_CLEANED_TWEETS_FILE = 'dataset/context_tweets_distance.csv'
# POLARIZATION_WORDS_DICTIONARY = 'dictionaries/lang_online_polarization_dict.csv'
# OUTPUT_FILE = 'outputs/polar_{0}.csv'.format(time.time())

ft = fasttext.load_model(MODEL_PATH)

all_tweets = pd.read_csv(CONTEXT_CLEANED_TWEETS_FILE, converters={"clean_text": literal_eval})[0:10]

start = time.time()
polarization_dictionary = pd.read_csv(POLARIZATION_WORDS_DICTIONARY)
polarization_words = set(polarization_dictionary.word.values)
polarization_embeddings = [ft.get_word_vector(word).reshape(1, -1) for word in polarization_words]

all_tweets['polar_words'] = all_tweets['clean_text'].apply(lambda text: polar_words(text, polarization_embeddings))
all_tweets['polar_words_n'] = all_tweets['polar_words'].str.split().map(len)
all_tweets['polar_words_ratio'] = all_tweets['polar_words_n'].astype('int') / len(all_tweets['clean_text'])

all_tweets = all_tweets.drop(columns=['clean_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)

print('FINISHED after {} seconds'.format(time.time()-start))
print(already_checked)