# Use this file to clean text to then calculate pattern-matching

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
import spacy
import time

CONTEXT_TWEETS_FILE = '../dataset/context_tweets.csv'
OUTPUT_FILE = '../outputs/context_tweets_pattern_matching_{0}.csv'.format(time.time())


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

df = pd.read_csv(CONTEXT_TWEETS_FILE)

df['stem_text'] = df.text.str.replace("amp;", "").str.replace('\n', '').str.replace("  ", "").str.strip().str.lower()

df = remove_text(df, 'stem_text')

df.stem_text = df.stem_text.apply(lambda x: re.sub(r'\W+', ' ', str(x)))

df['stem_text'] = df.stem_text.apply(lambda x: list(w for w in str(x).split() if len(w) > 3))
nlp = spacy.load('en_core_web_sm')

stop_words = nlp.Defaults.stop_words

stemmer = SnowballStemmer(language='english')

df['stem_text'] = df.stem_text.apply(lambda x: " ".join(stemmer.stem(w) for w in x if not (w in stop_words)))

df.to_csv(OUTPUT_FILE, index=False)
