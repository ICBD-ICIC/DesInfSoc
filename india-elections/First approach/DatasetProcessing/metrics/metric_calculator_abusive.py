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
ABUSIVE_WORDS_DICTIONARY = 'dictionaries/abuseLexicon.xlsx'

start = 0
steps = 1000000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
abuse = pd.read_excel(ABUSIVE_WORDS_DICTIONARY)
abus_words = abuse.word.str.lower().values
abus_words = list(stemmer.stem(w) for w in abus_words)
abus_words = set(abus_words)

while len(all_data[start:end]) != 0:
    df = all_data[start:end]

    df['abusive_words'] = df['stem_text'].apply(lambda x: " ".join(set(x.split()).intersection(abus_words)))
    df['abusive_words_n'] = df['abusive_words'].str.split().map(len)
    df['abusive_words_ratio'] = df['abusive_words_n'].astype('int') / df['stem_text'].str.split().map(len)

    num_cols = df.describe().columns
    df[num_cols] = df[num_cols].fillna(0).round(4)

    output_file = '../outputs/itrust/abusive/itrust_metrics_abusive_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)
    start = end
    end = start + steps
