import pandas as pd
import time
import sys
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
import transformers

DATASET_FILE = 'datasets/india-election-tweets-formatted-filtered-clean.csv'
INTERVAL_SIZE = 250000

start = 0
steps = 10000
end = start + steps

interval_init = INTERVAL_SIZE * int(sys.argv[1])
interval_end = interval_init + INTERVAL_SIZE

all_data = pd.read_csv(DATASET_FILE).iloc[interval_init:interval_end]
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

transformers.logging.set_verbosity(transformers.logging.ERROR)
analyzer = create_analyzer(task="sentiment", lang="en")

while len(all_data[start:end]) != 0:
    time_start = time.time()

    df = all_data[start:end]

    sequence = list(map(lambda text: preprocess_tweet(text, lang="en"), df.text.astype('str').tolist()))
    results = analyzer.predict(sequence)

    sentiments = pd.DataFrame(columns=['sentiment-positive', 'sentiment-negative', 'sentiment-neutral'])

    for index, result in enumerate(results):
        sentiments.loc[index] = {'sentiment-positive': result.probas['POS'],
                                 'sentiment-negative': result.probas['NEG'],
                                 'sentiment-neutral': result.probas['NEU']}

    output_file = 'datasets/sentiments/itrust_metrics_sentiments_interval_{0}-{1}_{2}.csv'.format(interval_init, interval_end, time.time())
    sentiments.index = df.index
    df = pd.concat([df, sentiments], axis=1).drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file)

    start = end
    end = start + steps

    print('Seconds to run {0} rows: {1}'.format(end - start, time.time() - time_start))

