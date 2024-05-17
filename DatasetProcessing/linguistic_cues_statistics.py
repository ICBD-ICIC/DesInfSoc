import pandas as pd

OUTPUT_FILE_PATH = 'linguistic_cues_statistics.csv'

ALL_TWEETS = pd.read_csv('dataset/india-election-tweets-metrics.csv')

LINGUISTIC_CUES = ['abusive_words_n',
                   'abusive_words_ratio',
                   'polar_words_n',
                   'polar_words_ratio',
                   'virtue_n',
                   'virtue_ratio',
                   'vice_n',
                   'vice_ratio',
                   'positive_words_n',
                   'positive_words_ratio',
                   'negative_words_n',
                   'negative_words_ratio']

statistics = []

for linguistic_cue in LINGUISTIC_CUES:
    non_zero_tweets = ALL_TWEETS[ALL_TWEETS[linguistic_cue] != 0]
    statistics.append({
        'linguistic_feature': linguistic_cue,
        'mean': non_zero_tweets[linguistic_cue].mean(),
        'std': non_zero_tweets[linguistic_cue].std()
    })

statistics_dataframe = pd.DataFrame(statistics)
statistics_dataframe.to_csv(OUTPUT_FILE_PATH)
