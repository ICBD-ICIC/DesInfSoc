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
MORAL_FOUNDATION_DICTIONARY = 'dictionaries/mfd.tsv'

start = 0
steps = 1000000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

stemmer = SnowballStemmer(language='english')
mfd = pd.read_csv(MORAL_FOUNDATION_DICTIONARY, sep='\t')
mfd.Word = mfd.Word.apply(lambda x: stemmer.stem(str(x).lower()))
mfd = mfd.drop_duplicates()

num_moral = ['num_mfd_care_virtue', 'num_mfd_care_vice', 'num_mfd_fairness_virtue',
             'num_mfd_fairness_vice', 'num_mfd_loyalty_virtue',
             'num_mfd_loyalty_vice', 'num_mfd_authority_virtue', 'num_mfd_authority_vice',
             'num_mfd_sanctity_virtue', 'num_mfd_sanctity_vice']

while len(all_data[start:end]) != 0:
    df = all_data[start:end]

    for c in mfd.Category.unique():
        words_mfd = set(mfd[mfd.Category == c]['Word'].values)
        df['mfd_' + c] = df.stem_text.apply(lambda x: " ".join(set(str(x).split()).intersection(words_mfd)))

    for c in num_moral:
        df[c] = df[c.replace('num_', '')].str.split().map(len)

    df['moral_words_n'] = df[num_moral].sum(axis=1)
    df['moral_words_ratio'] = df['moral_words_n'].astype('int') / df['stem_text'].str.split().map(len)

    output_file = 'outputs/itrust/mfd/itrust_metrics_mfd_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)
    start = end
    end = start + steps



