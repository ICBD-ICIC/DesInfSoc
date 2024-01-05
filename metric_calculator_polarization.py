## PREREQUISITES
## python -m spacy download en

import pandas as pd
import warnings
from nltk.stem.snowball import SnowballStemmer
import time

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = 'dataset/india-election-tweets-formatted-filtered-clean.csv'
POLARIZATION_WORDS_DICTIONARY = 'dictionaries/lang_online_polarization_dict.csv'

start = 0
steps = 1000000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
pdict = pd.read_csv(POLARIZATION_WORDS_DICTIONARY)
pol_wrd = pdict.word.values
pol_wrd = list(stemmer.stem(w.lower()) for w in pol_wrd)
pol_wrd = set(pol_wrd)

while len(all_data[start:end]) != 0:
    df = all_data[start:end]

    df['polar_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(pol_wrd)))
    df['polar_words_n'] = df['polar_words'].str.split().map(len)
    df['polar_words_ratio'] = df['polar_words_n'].astype('int') / df['stem_text'].str.split().map(len)

    output_file = 'outputs/itrust/polarization/itrust_metrics_polarization_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)
    start = end
    end = start + steps
