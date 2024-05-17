## PREREQUISITES
## python -m spacy download en

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time
import warnings

warnings.filterwarnings('ignore')

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

DATASET_FILE = '../dataset/india-election-tweets-formatted-missing-clean.csv'
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'

start = 0
steps = 2000000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
mfd = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')
mfd.Word = mfd.Word.apply(lambda x: stemmer.stem(str(x).lower()))
mfd = mfd.drop_duplicates()

while len(all_data[start:end]) != 0:
    df = all_data[start:end]

    for category in ['vice', 'virtue']:
        words_mfd = set(mfd[mfd["Category"].str.contains(category)]['Word'].values)
        df['mfd_' + category] = df.stem_text.apply(lambda x: " ".join(set(str(x).split()).intersection(words_mfd)))
        df[category + '_n'] = df['mfd_' + category].str.split().map(len)
        df[category + '_ratio'] = df[category + '_n'].astype('int') / df['stem_text'].str.split().map(len)

    output_file = '../outputs/itrust/mfd_binary/mfd_binary_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)

    start = end
    end = start + steps


