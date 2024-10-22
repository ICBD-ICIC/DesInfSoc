import pandas as pd

experiment_type = 'pattern_matching'

OUTPUT_FILE = '../outputs/tweets_features.csv'

ABUSIVE = pd.read_csv('../outputs/tweets/{}/abusive.csv'.format(experiment_type))
MFD = pd.read_csv('../outputs/tweets/{}/mfd.csv'.format(experiment_type))
POLAR = pd.read_csv('../outputs/tweets/{}/polar.csv'.format(experiment_type))
VALENCE = pd.read_csv('../outputs/tweets/{}/valence.csv'.format(experiment_type))
EMOTION = pd.read_csv('../outputs/tweets/emotion.csv')
SENTIMENT = pd.read_csv('../outputs/tweets/sentiment.csv')

tweets_features = pd.merge(ABUSIVE, MFD, on='id', how='outer')
tweets_features = pd.merge(tweets_features, POLAR, on='id', how='outer')
tweets_features = pd.merge(tweets_features, VALENCE, on='id', how='outer')
tweets_features = pd.merge(tweets_features, EMOTION, on='id', how='outer')
tweets_features = pd.merge(tweets_features, SENTIMENT, on='id', how='outer')

tweets_features.to_csv(OUTPUT_FILE)