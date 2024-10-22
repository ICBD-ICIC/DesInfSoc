from scipy.stats import norm
import json

with open('std_and_mean_pattern_matching.json', 'r') as file:
    STDS_MEANS = json.load(file)

EMOTIONS_CATEGORIES = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#                       0           1         2          3       4         5           6

SENTIMENTS_CATEGORIES = ['sentiment-neutral', 'sentiment-positive', 'sentiment-negative']
#                               0                        1                   2


# INTERVAL MAP
# [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
# 0         1             2           3
def discretize_percentage(value):
    return min(int(value * 4), 3)


def discretize_column_values(tweets, column_name, feature_name):
    if len(tweets.index) == 0:
        return 0
    mean = STDS_MEANS['{}_{}_mean'.format(feature_name, column_name)]
    std = STDS_MEANS['{}_{}_std'.format(feature_name, column_name)]
    amount_average = tweets[column_name].mean()
    density = norm.cdf(amount_average, loc=mean, scale=std)
    return discretize_percentage(density)


# 0: x = y
# 1: x > y
# 2: x < y
def discretize_comparison(x, y):
    if x == y:
        return 0
    elif x > y:
        return 1
    else:
        return 2


def discretize_abusive(tweets):
    abusive_amount_interval = discretize_column_values(tweets, 'abusive_words_n', 'abusive')
    abusive_ratio_interval = discretize_column_values(tweets, 'abusive_words_ratio', 'abusive')
    return abusive_amount_interval, abusive_ratio_interval


def discretize_polarization(tweets):
    polarization_amount_interval = discretize_column_values(tweets, 'polar_words_n', 'polar')
    polarization_ratio_interval = discretize_column_values(tweets, 'polar_words_ratio', 'polar')
    return polarization_amount_interval, polarization_ratio_interval


def discretize_mfd(tweets):
    mfd_virtue_amount = discretize_column_values(tweets, 'virtue_n', 'mdf')
    mfd_virtue_ratio = discretize_column_values(tweets, 'virtue_ratio', 'mdf')
    mfd_vice_amount = discretize_column_values(tweets, 'vice_n', 'mdf')
    mfd_vice_ratio = discretize_column_values(tweets, 'vice_ratio', 'mdf')
    return mfd_virtue_amount, mfd_virtue_ratio, mfd_vice_amount, mfd_vice_ratio


# 0: virtue = vice
# 1: virtue > vice
# 2: virtue < vice
def prediction_mfd(tweets):
    mfd = discretize_mfd(tweets)  # mfd_virtue_amount, mfd_virtue_ratio, mfd_vice_amount, mfd_vice_ratio
    mfd_ratio = discretize_comparison(mfd[1], mfd[3])
    mfd_amount = discretize_comparison(mfd[0], mfd[2])
    return mfd_amount, mfd_ratio


def discretize_valence(tweets):
    valence_positive_amount = discretize_column_values(tweets, 'positive_words_n', 'valence')
    valence_positive_ratio = discretize_column_values(tweets, 'positive_words_ratio','valence')
    valence_negative_amount = discretize_column_values(tweets, 'negative_words_n','valence')
    valence_negative_ratio = discretize_column_values(tweets, 'negative_words_ratio','valence')
    return valence_positive_amount, valence_positive_ratio, valence_negative_amount, valence_negative_ratio


# 0: positive = negative
# 1: positive > negative
# 2: positive < negative
def prediction_valence(tweets):
    valence = discretize_valence(tweets)  # valence_positive_amount, valence_positive_ratio, valence_negative_amount, valence_negative_ratio
    valence_ratio = discretize_comparison(valence[1], valence[3])
    valence_amount = discretize_comparison(valence[0], valence[2])
    return valence_amount, valence_ratio


# Default is 0
def predominant_category(tweets, categories):
    if len(tweets.index) == 0:
        return 0
    total_amount_per_category = list(map(lambda category: tweets[category].sum(), categories))
    predominant_category = max(total_amount_per_category)
    return total_amount_per_category.index(predominant_category)


def predominant_emotion(tweets):
    predominant_emotion = predominant_category(tweets, EMOTIONS_CATEGORIES)
    return (predominant_emotion,)


def predominant_sentiment(tweets):
    predominant_sentiment = predominant_category(tweets, SENTIMENTS_CATEGORIES)
    return (predominant_sentiment,)


def discretize_categories(tweets, all_categories):
    predominant_categories = tweets[all_categories].idxmax(axis=1)
    categories_total = predominant_categories.value_counts()
    total_tweets_amount = len(tweets.index)
    categories_percentage = categories_total.apply(
        lambda category_total: discretize_percentage(category_total / total_tweets_amount))

    categories_amounts = {}
    for category in all_categories:
        if category in categories_percentage.keys():
            categories_amounts[category] = categories_percentage[category]
        else:
            categories_amounts[category] = 0
    return tuple(categories_amounts[key] for key in all_categories)


def discretize_emotions(tweets):
    return discretize_categories(tweets, EMOTIONS_CATEGORIES)


def discretize_sentiments(tweets):
    return discretize_categories(tweets, SENTIMENTS_CATEGORIES)
