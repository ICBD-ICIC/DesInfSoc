from scipy.stats import norm

ABUSIVE_AMOUNT_MEAN = 0.10702764477410749
ABUSIVE_AMOUNT_DEVIATION = 0.3427292211597213
ABUSIVE_RATIO_MEAN = 0.013878248870572424
ABUSIVE_RATIO_DEVIATION = 0.04915942808971903

POLARIZATION_AMOUNT_MEAN = 0.3227449630764257
POLARIZATION_AMOUNT_DEVIATION = 0.5953114534859781
POLARIZATION_RATIO_MEAN = 0.04116681048357415
POLARIZATION_RATIO_DEVIATION = 0.08109465282505017

MFD_VICE_AMOUNT_MEAN = 0.1824626892761546
MFD_VICE_AMOUNT_DEVIATION = 0.44720561427572014
MFD_VICE_RATIO_MEAN = 0.023256898824071653
MFD_VICE_RATIO_DEVIATION = 0.06142892074228623
MFD_VIRTUE_AMOUNT_MEAN = 0.41211798998759724
MFD_VIRTUE_AMOUNT_DEVIATION = 0.6540318578939741
MFD_VIRTUE_RATIO_MEAN = 0.05244393412356305
MFD_VIRTUE_RATIO_DEVIATION = 0.08949401967338455

VALENCE_POSITIVE_AMOUNT_MEAN = 1.2712786871138455
VALENCE_POSITIVE_AMOUNT_DEVIATION = 1.1603875798860945
VALENCE_POSITIVE_RATIO_MEAN = 0.16419792907010802
VALENCE_POSITIVE_RATIO_DEVIATION = 0.15448028055614837
VALENCE_NEGATIVE_AMOUNT_MEAN = 1.0355423580213619
VALENCE_NEGATIVE_AMOUNT_DEVIATION = 1.099979369329647
VALENCE_NEGATIVE_RATIO_MEAN = 0.13304868421776658
VALENCE_NEGATIVE_RATIO_DEVIATION = 0.14619110840600053

EMOTIONS_CATEGORIES = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#                       0           1         2          3       4         5           6

SENTIMENTS_CATEGORIES = ['sentiment-neutral', 'sentiment-positive', 'sentiment-negative']
#                               0                        1                   2


# INTERVAL MAP
# [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
# 0         1             2           3
def discretize_percentage(value):
    return min(int(value * 4), 3)


def discretize_column_values(tweets, column_name, mean, std):
    if len(tweets.index) == 0:
        return 0
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
    abusive_amount_interval = discretize_column_values(tweets, 'abusive_words_n', ABUSIVE_AMOUNT_MEAN,
                                                       ABUSIVE_AMOUNT_DEVIATION)
    abusive_ratio_interval = discretize_column_values(tweets, 'abusive_words_ratio', ABUSIVE_RATIO_MEAN,
                                                      ABUSIVE_RATIO_DEVIATION)
    return abusive_amount_interval, abusive_ratio_interval


def discretize_polarization(tweets):
    polarization_amount_interval = discretize_column_values(tweets, 'polar_words_n',
                                                            POLARIZATION_AMOUNT_MEAN, POLARIZATION_AMOUNT_DEVIATION)
    polarization_ratio_interval = discretize_column_values(tweets, 'polar_words_ratio',
                                                           POLARIZATION_RATIO_MEAN, POLARIZATION_RATIO_DEVIATION)
    return polarization_amount_interval, polarization_ratio_interval


def discretize_mfd(tweets):
    mfd_virtue_amount = discretize_column_values(tweets, 'virtue_n', MFD_VIRTUE_AMOUNT_MEAN,
                                                 MFD_VIRTUE_AMOUNT_DEVIATION)
    mfd_virtue_ratio = discretize_column_values(tweets, 'virtue_ratio', MFD_VIRTUE_RATIO_MEAN,
                                                MFD_VIRTUE_RATIO_DEVIATION)
    mfd_vice_amount = discretize_column_values(tweets, 'vice_n', MFD_VICE_AMOUNT_MEAN,
                                               MFD_VICE_AMOUNT_DEVIATION)
    mfd_vice_ratio = discretize_column_values(tweets, 'vice_ratio', MFD_VICE_RATIO_MEAN,
                                              MFD_VICE_RATIO_DEVIATION)
    return mfd_virtue_amount, mfd_virtue_ratio, mfd_vice_amount, mfd_vice_ratio


# 0: virtue = vice
# 1: virtue > vice
# 2: virtue < vice
def prediction_mfd(tweets):
    mfd = discretize_mfd(tweets)  # mfd_virtue_amount, mfd_virtue_ratio, mfd_vice_amount, mfd_vice_ratio
    mfd_ratio = discretize_comparison(mfd[1], mfd[3])
    mfd_amount = discretize_comparison(mfd[0], mfd[2])
    return mfd_ratio, mfd_amount


def discretize_valence(tweets):
    valence_positive_amount = discretize_column_values(tweets, 'positive_words_n',
                                                       VALENCE_POSITIVE_AMOUNT_MEAN, VALENCE_POSITIVE_AMOUNT_DEVIATION)
    valence_positive_ratio = discretize_column_values(tweets, 'positive_words_ratio',
                                                      VALENCE_POSITIVE_RATIO_MEAN, VALENCE_POSITIVE_RATIO_DEVIATION)
    valence_negative_amount = discretize_column_values(tweets, 'negative_words_n',
                                                       VALENCE_NEGATIVE_AMOUNT_MEAN, VALENCE_NEGATIVE_AMOUNT_DEVIATION)
    valence_negative_ratio = discretize_column_values(tweets, 'negative_words_ratio',
                                                      VALENCE_NEGATIVE_RATIO_MEAN, VALENCE_NEGATIVE_RATIO_DEVIATION)
    return valence_positive_amount, valence_positive_ratio, valence_negative_amount, valence_negative_ratio


# 0: positive = negative
# 1: positive > negative
# 2: positive < negative
def prediction_valence(tweets):
    valence = discretize_valence(tweets)  # valence_positive_amount, valence_positive_ratio, valence_negative_amount, valence_negative_ratio
    valence_ratio = discretize_comparison(valence[1], valence[3])
    valence_amount = discretize_comparison(valence[0], valence[2])
    return valence_ratio, valence_amount


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
