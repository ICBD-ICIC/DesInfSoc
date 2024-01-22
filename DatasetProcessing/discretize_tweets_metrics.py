import pandas as pd
from scipy.stats import norm

# # ALL_TWEETS = '../dataset/india-election-tweets-formatted-filtered-clean.csv'
# ALL_TWEETS = '../outputs/test.csv'

ABUSIVE = pd.read_csv('dataset/itrust/abusive.csv').set_index('id')
ABUSIVE_AMOUNT_MEAN = ABUSIVE['abusive_words_n'].mean()
ABUSIVE_AMOUNT_DEVIATION = ABUSIVE['abusive_words_n'].std()
ABUSIVE_RATIO_MEAN = ABUSIVE['abusive_words_ratio'].mean()
ABUSIVE_RATIO_DEVIATION = ABUSIVE['abusive_words_ratio'].std()

POLARIZATION = pd.read_csv('dataset/itrust/polarization.csv').set_index('id')
POLARIZATION_AMOUNT_MEAN = POLARIZATION['polar_words_n'].mean()
POLARIZATION_AMOUNT_DEVIATION = POLARIZATION['polar_words_n'].std()
POLARIZATION_RATIO_MEAN = POLARIZATION['polar_words_ratio'].mean()
POLARIZATION_RATIO_DEVIATION = POLARIZATION['polar_words_ratio'].std()

MFD = pd.read_csv('dataset/itrust/mfd_binary.csv').set_index('id')
MFD_VICE_AMOUNT_MEAN = MFD['vice_n'].mean()
MFD_VICE_AMOUNT_DEVIATION = MFD['vice_n'].std()
MFD_VICE_RATIO_MEAN = MFD['vice_ratio'].mean()
MFD_VICE_RATIO_DEVIATION = MFD['vice_ratio'].std()
MFD_VIRTUE_AMOUNT_MEAN = MFD['virtue_n'].mean()
MFD_VIRTUE_AMOUNT_DEVIATION = MFD['virtue_n'].std()
MFD_VIRTUE_RATIO_MEAN = MFD['virtue_ratio'].mean()
MFD_VIRTUE_RATIO_DEVIATION = MFD['virtue_ratio'].std()

VALENCE = pd.read_csv('dataset/itrust/valence.csv').set_index('id')
VALENCE_POSITIVE_AMOUNT_MEAN = VALENCE['positive_words_n'].mean()
VALENCE_POSITIVE_AMOUNT_DEVIATION = VALENCE['positive_words_n'].std()
VALENCE_POSITIVE_RATIO_MEAN = VALENCE['positive_words_ratio'].mean()
VALENCE_POSITIVE_RATIO_DEVIATION = VALENCE['positive_words_ratio'].std()
VALENCE_NEGATIVE_AMOUNT_MEAN = VALENCE['negative_words_n'].mean()
VALENCE_NEGATIVE_AMOUNT_DEVIATION = VALENCE['negative_words_n'].std()
VALENCE_NEGATIVE_RATIO_MEAN = VALENCE['negative_words_ratio'].mean()
VALENCE_NEGATIVE_RATIO_DEVIATION = VALENCE['negative_words_ratio'].std()

EMOTIONS = pd.read_csv('dataset/itrust/emotions.csv').set_index('id')
EMOTIONS_CATEGORIES = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
#                       0           1       2       3       4           5           6

sentiments = pd.read_csv('dataset/itrust/sentiments.csv').set_index('id')
SENTIMENTS = sentiments.loc[:, ~sentiments.columns.str.contains('^Unnamed')]  # remove previous index
SENTIMENTS_CATEGORIES = ['sentiment-positive', 'sentiment-negative', 'sentiment-neutral']
#                               0                        1                   2

# OUTPUT_FILE = '../outputs/discretized_tweets.csv'

# INTERVAL MAP
# [0,25), [25,50), [50,75), [75,100]
# 0         1       2           3


def discretize_percentage(value):
    return int((value * 100) / 25)


def discretize_column_values(dataset, column_name, tweets_ids, mean, std):
    amount_average = dataset.loc[tweets_ids][column_name].sum()/len(tweets_ids)
    density = norm.cdf(amount_average, loc=mean, scale=std)
    return discretize_percentage(density)


def discretize_abusive(tweets_ids):
    abusive_amount_interval = discretize_column_values(ABUSIVE, 'abusive_words_n', tweets_ids, ABUSIVE_AMOUNT_MEAN,
                                                       ABUSIVE_AMOUNT_DEVIATION)
    abusive_ratio_interval = discretize_column_values(ABUSIVE, 'abusive_words_ratio', tweets_ids, ABUSIVE_RATIO_MEAN,
                                                      ABUSIVE_RATIO_DEVIATION)
    return {'abusive_amount_interval': abusive_amount_interval, 'abusive_ratio_interval': abusive_ratio_interval}


def discretize_polarization(tweets_ids):
    polarization_amount_interval = discretize_column_values(POLARIZATION, 'polar_words_n', tweets_ids,
                                                            POLARIZATION_AMOUNT_MEAN, POLARIZATION_AMOUNT_DEVIATION)
    polarization_ratio_interval = discretize_column_values(POLARIZATION, 'polar_words_ratio', tweets_ids,
                                                           POLARIZATION_RATIO_MEAN, POLARIZATION_RATIO_DEVIATION)
    return {'polarization_amount_interval': polarization_amount_interval,
            'polarization_ratio_interval': polarization_ratio_interval}


def discretize_mfd(tweets_ids):
    mfd_virtue_amount = discretize_column_values(MFD, 'virtue_n', tweets_ids, MFD_VIRTUE_AMOUNT_MEAN,
                                                 MFD_VIRTUE_AMOUNT_DEVIATION)
    mfd_virtue_ratio = discretize_column_values(MFD, 'virtue_ratio', tweets_ids, MFD_VIRTUE_RATIO_MEAN,
                                                MFD_VIRTUE_RATIO_DEVIATION)
    mfd_vice_amount = discretize_column_values(MFD, 'vice_n', tweets_ids, MFD_VICE_AMOUNT_MEAN,
                                               MFD_VICE_AMOUNT_DEVIATION)
    mfd_vice_ratio = discretize_column_values(MFD, 'vice_ratio', tweets_ids, MFD_VICE_RATIO_MEAN,
                                              MFD_VICE_RATIO_DEVIATION)
    return {'mfd_virtue_amount': mfd_virtue_amount,
            'mfd_virtue_ratio': mfd_virtue_ratio,
            'mfd_vice_amount': mfd_vice_amount,
            'mfd_vice_ratio': mfd_vice_ratio}


def discretize_valence(tweets_ids):
    valence_positive_amount = discretize_column_values(VALENCE, 'positive_words_n', tweets_ids,
                                                       VALENCE_POSITIVE_AMOUNT_MEAN, VALENCE_POSITIVE_AMOUNT_DEVIATION)
    valence_positive_ratio = discretize_column_values(VALENCE, 'positive_words_ratio', tweets_ids,
                                                      VALENCE_POSITIVE_RATIO_MEAN, VALENCE_POSITIVE_RATIO_DEVIATION)
    valence_negative_amount = discretize_column_values(VALENCE, 'negative_words_n', tweets_ids,
                                                       VALENCE_NEGATIVE_AMOUNT_MEAN, VALENCE_NEGATIVE_AMOUNT_DEVIATION)
    valence_negative_ratio = discretize_column_values(VALENCE, 'negative_words_ratio', tweets_ids,
                                                      VALENCE_NEGATIVE_RATIO_MEAN, VALENCE_NEGATIVE_RATIO_DEVIATION)
    return {'valence_positive_amount': valence_positive_amount,
            'valence_positive_ratio': valence_positive_ratio,
            'valence_negative_amount': valence_negative_amount,
            'valence_negative_ratio': valence_negative_ratio}


def predominant_category(dataset, categories, tweets_ids):
    tweets = dataset.loc[tweets_ids]
    total_amount_per_category = list(map(lambda category: tweets[category].sum(), categories))
    predominant_category = max(total_amount_per_category)
    return total_amount_per_category.index(predominant_category)


def predominant_emotion(tweets_ids):
    predominant_emotion = predominant_category(EMOTIONS, EMOTIONS_CATEGORIES, tweets_ids)
    return {'predominant_emotion': predominant_emotion}


def predominant_sentiment(tweets_ids):
    predominant_sentiment = predominant_category(SENTIMENTS, SENTIMENTS_CATEGORIES, tweets_ids)
    return {'predominant_sentiment': predominant_sentiment}


def discretize_categories(dataset, all_categories, tweets_ids):
    tweets = dataset.loc[tweets_ids]
    predominant_categories = tweets.idxmax(axis=1)
    categories_total = predominant_categories.value_counts()
    categories_percentage = categories_total.apply(lambda category_total: discretize_percentage(category_total/len(tweets_ids)))

    categories_amounts = {}
    for category in all_categories:
        if category in categories_percentage.keys():
            categories_amounts['average_' + category] = categories_percentage[category]
        else:
            categories_amounts['average_' + category] = 0
    return categories_amounts


def discretize_emotions(tweets_ids):
    return discretize_categories(EMOTIONS, EMOTIONS_CATEGORIES, tweets_ids)


def discretize_sentiments(tweets_ids):
    return discretize_categories(SENTIMENTS, SENTIMENTS_CATEGORIES, tweets_ids)


# all_tweets = pd.read_csv(ALL_TWEETS)
# all_tweets = all_tweets.loc[:, ~all_tweets.columns.str.contains('^Unnamed')]  # remove previous index
# all_tweets = all_tweets.set_index('id')
#
# tweets_ids = all_tweets.index.to_list()
#
# context = {
#     **discretize_abusive(tweets_ids),
#     **discretize_polarization(tweets_ids),
#     **predominant_emotion(tweets_ids),
#     **discretize_emotions(tweets_ids),
#     **discretize_mfd(tweets_ids),
#     **discretize_valence(tweets_ids),
#     **predominant_sentiment(tweets_ids),
#     **discretize_sentiments(tweets_ids)
# }
#
# print(context)
