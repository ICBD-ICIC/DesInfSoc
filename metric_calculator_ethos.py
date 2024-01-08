## PREREQUISITES
## python -m spacy download en

import pandas as pd
import warnings
from transformers import pipeline
import time

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

warnings.filterwarnings('ignore')

DATASET_FILE = 'dataset/india-election-tweets-formatted-filtered-clean.csv'

start = 0
steps = 100
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}

model_path = "eevvgg/Stance-Tw"
cls_task = pipeline(task="text-classification", model=model_path, tokenizer=model_path)  # , device=0

while len(all_data[start:end]) != 0:
    time_start = time.time()

    df = all_data[start:end]

    sequence = df.text.astype('str').tolist()
    result = cls_task(sequence, **tokenizer_kwargs)
    conf = list(x['score'] for x in result)
    labels = list(i['label'] for i in result)
    df['ethos'] = labels
    df['CF_ethos'] = conf
    df['CF_ethos'] = df['CF_ethos'].round(4)

    output_file = 'outputs/itrust/ethos/itrust_metrics_ethos_{0}.csv'.format(time.time())
    df = df.drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)
    start = end
    end = start + steps

    print('Seconds to run {0} rows: {1}'.format(end-start, time.time()-time_start))
