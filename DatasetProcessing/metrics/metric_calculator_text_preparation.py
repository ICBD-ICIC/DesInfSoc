# Use this file to clean text to analyze

import pandas as pd
import warnings
from nltk.stem.snowball import SnowballStemmer
import re
import spacy
import time

start = time.time()
interval = time.time()

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = '../dataset/outputs/india-election-tweets-formatted-missing.csv'
OUTPUT_FILE = '../dataset/outputs/india-election-tweets-formatted-missing-clean.csv'


def remove_text(df, text_column):
    new_texts = []
    for text in df[text_column]:
        text_list = str(text).split(" ")
        new_string_list = []
        for word in text_list:
            if 'http' in word:
                word = ""
            elif ('@' in word) and (len(word) > 1):
                word = ""
            elif ('#' in word) and (len(word) > 1):
                word = ""
            if not word.isnumeric():
                new_string_list.append(word)
        new_string = " ".join(new_string_list)
        new_string = new_string.strip()
        new_texts.append(new_string)
    df['stem_text'] = new_texts  # cln_text
    return df


print('Loading dataset...')

df = pd.read_csv(DATASET_FILE)

print('Finish loading - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

df['stem_text'] = df.text.str.replace("amp;", "").str.replace('\n', '').str.replace("  ", "").str.strip().str.lower()

print('Finish replace - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

df = remove_text(df, 'stem_text')

print('Finish remove_text - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

df.stem_text = df.stem_text.apply(lambda x: re.sub(r'\W+', ' ', str(x)))

print('Finish lambda sub - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

df['stem_text'] = df.stem_text.apply(lambda x: list(w for w in str(x).split() if len(w) > 3))

print('Finish lambda list - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

nlp = spacy.load('en_core_web_sm')

print('Finish spacy.load - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

stop_words = nlp.Defaults.stop_words

print('Finish stop_words - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

stemmer = SnowballStemmer(language='english')

print('Finish snowball - Process seconds: {0}'.format(time.time() - interval))
interval = time.time()

df['stem_text'] = df.stem_text.apply(lambda x: " ".join(stemmer.stem(w) for w in x if not (w in stop_words)))

print('Finish lambda join - Process seconds: {0}'.format(time.time() - interval))

df.to_csv(OUTPUT_FILE)

print('Total seconds: {0}'.format(time.time() - start))
