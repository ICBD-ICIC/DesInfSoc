# Use this file to calculate iTrust metrics
# Needs some refactoring

## PREREQUISITES
## python -m spacy download en

import pandas as pd
import warnings
from nltk.stem.snowball import SnowballStemmer
from transformers import pipeline
import time

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = 'dataset/india-election-tweets-formatted-filtered-clean.csv'


OUTPUT_FILE = 'outputs/itrust/itrust_metrics_{0}.csv'.format(time.time())
START = 0
END = 1

df = pd.read_csv(DATASET_FILE)[START:END]

stemmer = SnowballStemmer(language='english')


sequence = df.text.astype('str').tolist()
check = "j-hartmann/emotion-english-distilroberta-base"
sentiment_analysis = pipeline("sentiment-analysis", model=check)  # , device=0

result = sentiment_analysis(sequence)
df['emotion2'] = [x['label'] for x in result]
df['cf_emotion2'] = [x['score'] for x in result]
df['cf_emotion2'] = df['cf_emotion2'].round(4)

df.to_csv(OUTPUT_FILE)
