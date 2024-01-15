import pandas as pd
import time
from pysentimiento import create_analyzer
import transformers

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

DATASET_FILE = 'dataset/india-election-tweets-formatted-filtered-clean.csv'

start = 0
steps = 10000
end = start + steps

all_data = pd.read_csv(DATASET_FILE)
all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]  # remove previous index
all_data = all_data.astype({'stem_text': str})

transformers.logging.set_verbosity(transformers.logging.ERROR)
analyzer = create_analyzer(task="sentiment", lang="es")

while len(all_data[start:end]) != 0:
    time_start = time.time()

    df = all_data[start:end]

    sequence = df.text.astype('str').tolist()
    results = analyzer.predict(sequence)

    sentiments = pd.DataFrame(columns=['sentiment-positive', 'sentiment-negative', 'sentiment-neutral'])

    for index, result in enumerate(results):
        sentiments.loc[index] = {'sentiment-positive': result.probas['POS'],
                                 'sentiment-negative': result.probas['NEG'],
                                 'sentiment-neutral': result.probas['NEU']}

    output_file = 'outputs/sentiments/pysentimiento_sentiments_{0}.csv'.format(time.time())
    df = pd.concat([df, sentiments], axis=1).drop(columns=['user_id', 'text', 'stem_text', 'username', 'created_at'])
    df.to_csv(output_file, index=False)

    start = end
    end = start + steps

    print('Seconds to run {0} rows: {1}'.format(end - start, time.time() - time_start))
