from scipy.stats import norm
import pandas as pd

STATISTICS = pd.read_csv('linguistic_cues_statistics.csv').set_index('linguistic_feature')

EMOTIONS_CATEGORIES = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#                       0           1         2          3       4         5           6

SENTIMENTS_CATEGORIES = ['sentiment-neutral', 'sentiment-positive', 'sentiment-negative']
#                               0                        1                   2


# INTERVAL MAP
# [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
# 0         1             2           3
def discretize_percentage(value):
    return min(int(value * 4), 3)


def discretize_column_values(tweets, column_name):
    # When user takes no action, there are no tweets in the prediction interval
    if len(tweets.index) == 0:
        return 0
    amount_average = tweets[tweets[column_name] > 0][column_name].mean()
    mean = STATISTICS.loc[column_name, 'mean']
    std = STATISTICS.loc[column_name, 'std']
    density = norm.cdf(amount_average, loc=mean, scale=std)
    return discretize_percentage(density)


# 0: x = y
# 1: x > y
# 2: x < y
def discretize_comparison(x, y, has_x, has_y):
    if (not has_x and not has_y) or x == y:
        return 0
    elif not has_y or x > y:
        return 1
    else: # not has_x or x < y
        return 2


def has_words(tweets, column_name):
    if tweets[column_name].sum() > 0:
        return 1
    else:
        return 0


def discretize_abusive(tweets):
    abusive_words_present = has_words(tweets, 'abusive_words_n')
    abusive_amount_interval = abusive_ratio_interval = 0
    if abusive_words_present:
        abusive_amount_interval = discretize_column_values(tweets, 'abusive_words_n')
        abusive_ratio_interval = discretize_column_values(tweets, 'abusive_words_ratio')
    return abusive_amount_interval, abusive_ratio_interval, abusive_words_present


def discretize_polarization(tweets):
    polarization_words_present = has_words(tweets, 'polar_words_n')
    polarization_amount_interval = polarization_ratio_interval = 0
    if polarization_words_present:
        polarization_amount_interval = discretize_column_values(tweets, 'polar_words_n')
        polarization_ratio_interval = discretize_column_values(tweets, 'polar_words_ratio')
    return polarization_amount_interval, polarization_ratio_interval, polarization_words_present


def discretize_mfd(tweets):
    mdf_virtue_words_present = has_words(tweets, 'virtue_n')
    mfd_virtue_amount = mfd_virtue_ratio = 0
    if mdf_virtue_words_present:
        mfd_virtue_amount = discretize_column_values(tweets, 'virtue_n')
        mfd_virtue_ratio = discretize_column_values(tweets, 'virtue_ratio')
    mfd_vice_words_present = has_words(tweets, 'vice_n')
    mfd_vice_amount = mfd_vice_ratio = 0
    if mfd_vice_words_present:
        mfd_vice_amount = discretize_column_values(tweets, 'vice_n')
        mfd_vice_ratio = discretize_column_values(tweets, 'vice_ratio')
    return (mfd_virtue_amount, mfd_virtue_ratio, mdf_virtue_words_present,
            mfd_vice_amount, mfd_vice_ratio, mfd_vice_words_present)


# 0: virtue = vice
# 1: virtue > vice
# 2: virtue < vice
def prediction_mfd(tweets):
    (mfd_virtue_amount, mfd_virtue_ratio, has_virtue,
     mfd_vice_amount, mfd_vice_ratio, has_vice) = discretize_mfd(tweets)
    mfd_amount = discretize_comparison(mfd_virtue_amount, mfd_vice_amount, has_virtue, has_vice)
    mfd_ratio = discretize_comparison(mfd_virtue_ratio, mfd_vice_ratio, has_virtue, has_vice)
    return mfd_amount, mfd_ratio, has_virtue, has_vice


def discretize_valence(tweets):
    positive_words_present = has_words(tweets, 'positive_words_n')
    valence_positive_amount = valence_positive_ratio = 0
    if positive_words_present:
        valence_positive_amount = discretize_column_values(tweets, 'positive_words_n')
        valence_positive_ratio = discretize_column_values(tweets, 'positive_words_ratio')
    negative_words_present = has_words(tweets, 'negative_words_n')
    valence_negative_amount = valence_negative_ratio = 0
    if negative_words_present:
        valence_negative_amount = discretize_column_values(tweets, 'negative_words_n')
        valence_negative_ratio = discretize_column_values(tweets, 'negative_words_ratio')
    return (valence_positive_amount, valence_positive_ratio, positive_words_present,
            valence_negative_amount, valence_negative_ratio, negative_words_present)


# 0: positive = negative
# 1: positive > negative
# 2: positive < negative
def prediction_valence(tweets):
    (valence_positive_amount, valence_positive_ratio, has_positive,
     valence_negative_amount, valence_negative_ratio, has_negative) = discretize_valence(tweets)
    valence_amount = discretize_comparison(valence_positive_amount, valence_negative_amount, has_positive, has_negative)
    valence_ratio = discretize_comparison(valence_positive_ratio, valence_negative_ratio, has_positive, has_negative)
    return valence_amount, valence_ratio, has_positive, has_negative


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
