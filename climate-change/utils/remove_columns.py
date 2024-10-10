# Keeps useful columns
# Fix encoding issues
# Saves as csv instead of xslx

import pandas as pd
import ftfy

DATASETS = ['../dataset/replies.xlsx',
            '../dataset/profiles.xlsx',
            '../dataset/influencers.xlsx']

for dataset in DATASETS:
    df = pd.read_excel(dataset)
    try:
        df = df[['id','created_at','text','conversation_id','username']]
    except KeyError:
        df = df[['id', 'created_at', 'text', 'username']]
    df['text'] = df['text'].apply(lambda text: ftfy.fix_text(text))
    df.to_csv(dataset.replace('.xlsx','.csv'), index=False)

