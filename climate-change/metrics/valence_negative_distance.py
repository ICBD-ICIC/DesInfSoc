# Calculates amount and ratio of valence negative words using distance words embeddings with a dictionary

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
def valence_negative_words(text, dictionary):
    negative_or_similar = []
    for word in text:
        found = False
        if word in already_found_words:
            negative_or_similar.append(word)
            found = True
        elif not word in already_checked:
            word_embedding = ft.get_word_vector(word).reshape(1, -1)
            for dictionary_word in dictionary:
                if cosine_similarity(word_embedding, dictionary_word) >= 0.6:
                    negative_or_similar.append(word)
                    already_found_words.append(word)
                    found = True
                    break
        if not found:
            already_checked.append(word)
    return " ".join(negative_or_similar)

# Local paths
MODEL_PATH = '../../../crawl-300d-2M-subword.bin'
CONTEXT_CLEANED_TWEETS_FILE = '../dataset/context_tweets_distance.csv'
VALENCE_WORDS_DICTIONARY = 'dictionaries/anew_val_polarity.xlsx'
OUTPUT_FILE = '../outputs/valence_negative_{0}.csv'.format(time.time())

# Paths in the kluster
# MODEL_PATH = 'crawl-300d-2M-subword.bin'
# CONTEXT_CLEANED_TWEETS_FILE = 'dataset/context_tweets_distance.csv'
# VALENCE_WORDS_DICTIONARY = 'dictionaries/anew_val_polarity.xlsx'
# OUTPUT_FILE = 'outputs/valence_negative_{0}.csv'.format(time.time())

ft = fasttext.load_model(MODEL_PATH)

all_tweets = pd.read_csv(CONTEXT_CLEANED_TWEETS_FILE, converters={"clean_text": literal_eval})[0:10]

start = time.time()
valence_dictionary = pd.read_excel(VALENCE_WORDS_DICTIONARY)
valence_dictionary = valence_dictionary[(valence_dictionary.Valence_standardized < -0.5) | (valence_dictionary.Valence_standardized > 1)]
valence_dictionary['Word'] = valence_dictionary['Word'].astype('str').str.lower()
negative_words = valence_dictionary[(valence_dictionary.Valence_standardized < -0.5)]['Word'].tolist()
negative_words = set(negative_words)
negative_words.remove('nan')
negative_embeddings = [ft.get_word_vector(word).reshape(1, -1) for word in negative_words]

all_tweets['negative_words'] = all_tweets['clean_text'].apply(lambda text: valence_negative_words(text, negative_embeddings))
all_tweets['negative_words_n'] = all_tweets['negative_words'].str.split().map(len)
all_tweets['negative_words_ratio'] = all_tweets['negative_words_n'].astype('int') / all_tweets['clean_text'].map(len)

num_cols = all_tweets.describe().columns
all_tweets[num_cols] = all_tweets[num_cols].fillna(0).round(4)

all_tweets = all_tweets.drop(columns=['clean_text'])
all_tweets.to_csv(OUTPUT_FILE, index=False)

print('FINISHED after {} seconds'.format(time.time()-start))
