## PREREQUISITES
## python -m spacy download en

import pandas as pd
import warnings
from nltk.stem.snowball import SnowballStemmer
import time

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = '../dataset/india-election-tweets-formatted-missing-clean.csv'
VALENCE_WORDS_DICTIONARY = 'dictionaries/anew_val_polarity.xlsx'

start = 0
steps = 1000000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
affdf2 = pd.read_excel(VALENCE_WORDS_DICTIONARY)
affdf2 = affdf2[(affdf2.Valence_standardized < -0.5) | (affdf2.Valence_standardized > 1)]
affdf2['Word'] = affdf2['Word'].astype('str').str.lower()
neg_words = affdf2[(affdf2.Valence_standardized < -0.5)]['Word'].tolist()
pos_words = affdf2[(affdf2.Valence_standardized > 1)]['Word'].tolist()
neg_words.remove('nan')
neg_words = list(stemmer.stem(w) for w in neg_words)
pos_words = list(stemmer.stem(w) for w in pos_words)
neg_words = set(neg_words)
pos_words = set(pos_words)

while len(all_data[start:end]) != 0:
    df = all_data[start:end]

    df['negative_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(neg_words)))
    df['positive_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(pos_words)))

    df['negative_words_n'] = df['negative_words'].str.split().map(len)
    df['negative_words_ratio'] = df['negative_words'].str.split().map(len) / df['stem_text'].str.split().map(len)

    df['positive_words_n'] = df['positive_words'].str.split().map(len)
    df['positive_words_ratio'] = df['positive_words'].str.split().map(len) / df['stem_text'].str.split().map(len)

    output_file = '../outputs/itrust/valence/itrust_metrics_valence_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)
    start = end
    end = start + steps
