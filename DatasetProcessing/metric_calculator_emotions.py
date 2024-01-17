import pandas as pd
import sys
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DATASET_FILE = 'datasets/india-election-tweets-formatted-filtered-clean.csv'
INTERVAL_SIZE = 1000000

start = 0
steps = 10000
end = start + steps

interval_init = INTERVAL_SIZE * int(sys.argv[1])
interval_end = interval_init + INTERVAL_SIZE

all_data = pd.read_csv(DATASET_FILE).iloc[interval_init:interval_end]
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

while len(all_data[start:end]) != 0:
    time_start = time.time()

    df = all_data[start:end]

    sequence = df.text.astype('str').tolist()
    results = classifier(sequence)

    emotions = pd.DataFrame(columns=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])

    for index, result in enumerate(results):
        for item in result:
            emotions.loc[index, item['label']] = item['score']

    output_file = 'datasets/emotions/itrust_metrics_emotion_interval_{0}:{1}_{2}.csv'.format(interval_init, interval_end, time.time())
    emotions.index = df.index
    df = pd.concat([df, emotions], axis=1).drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file)
    start = end
    end = start + steps

    print('Seconds to run {0} rows: {1}'.format(end - start, time.time() - time_start))
