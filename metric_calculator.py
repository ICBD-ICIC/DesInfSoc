# Use this file to calculate iTrust metrics
# Needs some refactoring

## PREREQUISITES
## python -m spacy download en

import pandas as pd
import warnings
from nltk.stem.snowball import SnowballStemmer
import re
import spacy
from transformers import pipeline
import time

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = 'dataset/india-election-tweets-formatted.csv'
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'
POLARIZATION_WORDS_DICTIONARY = 'dictionaries/lang_online_polarization_dict.csv'
ABUSIVE_WORDS_DICTIONARY = 'dictionaries/abuseLexicon.xlsx'
VALENCE_WORDS_DICTIONARY = 'dictionaries/anew_val_polarity.xlsx'
OUTPUT_FILE = 'outputs/itrust/itrust_metrics_{0}.csv'.format(time.time())
START = 0
END = 10


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


df = pd.read_csv(DATASET_FILE)[START:END]

df['stem_text'] = df.text.str.replace("amp;", "").str.replace('\n', '').str.replace("  ", "").str.strip().str.lower()

df = remove_text(df, 'stem_text')

df.stem_text = df.stem_text.apply(lambda x: re.sub(r'\W+', ' ', str(x)))
df['stem_text'] = df.stem_text.apply(lambda x: list(w for w in str(x).split() if len(w) > 3))

nlp = spacy.load('en_core_web_sm')

stop_words = nlp.Defaults.stop_words

stemmer = SnowballStemmer(language='english')

df['stem_text'] = df.stem_text.apply(lambda x: " ".join(stemmer.stem(w) for w in x if not (w in stop_words)))

affdf2 = pd.read_excel(VALENCE_WORDS_DICTIONARY)

affdf2 = affdf2[(affdf2.Valence_standardized < -0.5) | (affdf2.Valence_standardized > 1)]
affdf2['Word'] = affdf2['Word'].astype('str').str.lower()

neg_words = affdf2[(affdf2.Valence_standardized < -0.5)]['Word'].tolist()
pos_words = affdf2[(affdf2.Valence_standardized > 1)]['Word'].tolist()
neg_words.remove('nan')

len(pos_words), len(neg_words)

neg_words = list(stemmer.stem(w) for w in neg_words)
pos_words = list(stemmer.stem(w) for w in pos_words)

neg_words = set(neg_words)
pos_words = set(pos_words)
len(pos_words), len(neg_words)

df['negative_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(neg_words)))
df['positive_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(pos_words)))

df['negative_words_n'] = df['negative_words'].str.split().map(len)
df['negative_words_ratio'] = df['negative_words'].str.split().map(len) / df['stem_text'].str.split().map(len)

df['positive_words_n'] = df['positive_words'].str.split().map(len)
df['positive_words_ratio'] = df['positive_words'].str.split().map(len) / df['stem_text'].str.split().map(len)

mfd = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')

mfd.Word = mfd.Word.apply(lambda x: stemmer.stem(str(x).lower()))

mfd = mfd.drop_duplicates()

for c in mfd.Category.unique():
    words_mfd = set(mfd[mfd.Category == c]['Word'].values)
    df['mfd_' + c] = df.stem_text.apply(lambda x: " ".join(set(str(x).split()).intersection(words_mfd)))

num_moral = ['num_mfd_care_virtue', 'num_mfd_care_vice', 'num_mfd_fairness_virtue',
             'num_mfd_fairness_vice', 'num_mfd_loyalty_virtue',
             'num_mfd_loyalty_vice', 'num_mfd_authority_virtue', 'num_mfd_authority_vice',
             'num_mfd_sanctity_virtue', 'num_mfd_sanctity_vice']

for c in num_moral:
    df[c] = df[c.replace('num_', '')].str.split().map(len)

df['moral_words_n'] = df[num_moral].sum(axis=1)
df['moral_words_ratio'] = df['moral_words_n'].astype('int') / df['stem_text'].str.split().map(len)

pdict = pd.read_csv(POLARIZATION_WORDS_DICTIONARY)

pol_wrd = pdict.word.values
pol_wrd = list(stemmer.stem(w.lower()) for w in pol_wrd)

pol_wrd = set(pol_wrd)

df['polar_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(pol_wrd)))
df['polar_words_n'] = df['polar_words'].str.split().map(len)
df['polar_words_ratio'] = df['polar_words_n'].astype('int') / df['stem_text'].str.split().map(len)

abuse = pd.read_excel(ABUSIVE_WORDS_DICTIONARY)

abus_words = abuse.word.str.lower().values
abus_words = list(stemmer.stem(w) for w in abus_words)
abus_words = set(abus_words)

df['abusive_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(abus_words)))
df['abusive_words_n'] = df['abusive_words'].str.split().map(len)
df['abusive_words_ratio'] = df['abusive_words_n'].astype('int') / df['stem_text'].str.split().map(len)

num_cols = df.describe().columns
df[num_cols] = df[num_cols].fillna(0).round(4)

# df['public_metrics'] = df['public_metrics'].astype('str').fillna("{}").apply(literal_eval)
# df['public_metrics_user'] = df['public_metrics_user'].astype('str').fillna("{}").apply(literal_eval)
# df['followers_count'] = df.public_metrics_user.apply(lambda x: x['followers_count'])
# df['following_count'] = df.public_metrics_user.apply(lambda x: x['following_count'])
# df['tweet_count'] = df.public_metrics_user.apply(lambda x: x['tweet_count'])

# df['retweet_count'] = df.public_metrics.apply(lambda x: x['retweet_count'])
# df['reply_count'] = df.public_metrics.apply(lambda x: x['reply_count'])
# df['like_count'] = df.public_metrics.apply(lambda x: x['like_count'])
# df['quote_count'] = df.public_metrics.apply(lambda x: x['quote_count'])
# df['impression_count'] = df.public_metrics.apply(lambda x: x['impression_count'])

sequence = df.text.astype('str').tolist()
check = "j-hartmann/emotion-english-distilroberta-base"
sentiment_analysis = pipeline("sentiment-analysis", model=check)  # , device=0

result = sentiment_analysis(sequence)
df['emotion2'] = [x['label'] for x in result]
df['cf_emotion2'] = [x['score'] for x in result]
df['cf_emotion2'] = df['cf_emotion2'].round(4)

tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}

model_path = "eevvgg/Stance-Tw"
cls_task = pipeline(task="text-classification", model=model_path, tokenizer=model_path)  # , device=0
result = cls_task(sequence, **tokenizer_kwargs)

conf = list(x['score'] for x in result)
labels = list(i['label'] for i in result)

df['ethos'] = labels
df['CF_ethos'] = conf
df['CF_ethos'] = df['CF_ethos'].round(4)

df.to_csv(OUTPUT_FILE)
