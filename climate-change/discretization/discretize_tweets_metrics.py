# Normalizes and discretizes the tweets features depending on the experiment_type

from scipy.stats import norm
import json
import os


# INTERVAL MAP
# [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
# 0         1             2           3
def discretize_percentage(value):
    return min(int(value * 4), 3)

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
    return categories_amounts

# Default is 0
def predominant_category(tweets, categories):
    if len(tweets.index) == 0:
        return 0
    total_amount_per_category = list(map(lambda category: tweets[category].sum(), categories))
    predominant_category = max(total_amount_per_category)
    return total_amount_per_category.index(predominant_category)

class TweetsMetricsDiscretizer:

    def __init__(self, experiment_type):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        std_and_mean = os.path.join(current_dir, 'std_and_mean_{}.json'.format(experiment_type))
        with open(std_and_mean, 'r') as file:
            self.std_means = json.load(file)
        self.emotions_categories = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        #                       0           1         2          3       4         5           6
        self.sentiment_categories = ['sentiment-neutral', 'sentiment-positive', 'sentiment-negative']
        #                               0                        1                   2

    def discretize_column_values(self, tweets, column_name, feature_name):
        if len(tweets.index) == 0:
            return 0
        mean = self.std_means['{}_{}_mean'.format(feature_name, column_name)]
        std = self.std_means['{}_{}_std'.format(feature_name, column_name)]
        amount_average = tweets[column_name].mean()
        density = norm.cdf(amount_average, loc=mean, scale=std)
        return discretize_percentage(density)

    def discretize_abusive(self, tweets):
        abusive_amount_interval = self.discretize_column_values(tweets, 'abusive_words_n', 'abusive')
        abusive_ratio_interval = self.discretize_column_values(tweets, 'abusive_words_ratio', 'abusive')
        return {'abusive_amount_interval': abusive_amount_interval,
                'abusive_ratio_interval': abusive_ratio_interval}

    def discretize_polarization(self, tweets):
        polarization_amount_interval = self.discretize_column_values(tweets, 'polar_words_n', 'polar')
        polarization_ratio_interval = self.discretize_column_values(tweets, 'polar_words_ratio', 'polar')
        return {'polarization_amount_interval': polarization_amount_interval,
                'polarization_ratio_interval': polarization_ratio_interval}

    def discretize_mfd(self, tweets):
        mfd_virtue_amount = self.discretize_column_values(tweets, 'virtue_n', 'mfd')
        mfd_virtue_ratio = self.discretize_column_values(tweets, 'virtue_ratio', 'mfd')
        mfd_vice_amount = self.discretize_column_values(tweets, 'vice_n', 'mfd')
        mfd_vice_ratio = self.discretize_column_values(tweets, 'vice_ratio', 'mfd')
        return {'mfd_virtue_amount': mfd_virtue_amount,
                'mfd_virtue_ratio': mfd_virtue_ratio,
                'mfd_vice_amount': mfd_vice_amount,
                'mfd_vice_ratio': mfd_vice_ratio}

    # 0: virtue = vice
    # 1: virtue > vice
    # 2: virtue < vice
    def prediction_mfd(self, tweets):
        mfd = self.discretize_mfd(tweets)
        mfd_ratio = discretize_comparison(mfd['mfd_virtue_ratio'], mfd['mfd_vice_ratio'])
        mfd_amount = discretize_comparison(mfd['mfd_virtue_amount'], mfd['mfd_vice_amount'])
        return {'mfd_amount': mfd_amount,
                'mfd_ratio': mfd_ratio}

    def discretize_valence(self, tweets):
        valence_positive_amount = self.discretize_column_values(tweets, 'positive_words_n', 'valence')
        valence_positive_ratio = self.discretize_column_values(tweets, 'positive_words_ratio', 'valence')
        valence_negative_amount = self.discretize_column_values(tweets, 'negative_words_n', 'valence')
        valence_negative_ratio = self.discretize_column_values(tweets, 'negative_words_ratio', 'valence')
        return {'valence_positive_amount': valence_positive_amount,
                'valence_positive_ratio': valence_positive_ratio,
                'valence_negative_amount': valence_negative_amount,
                'valence_negative_ratio': valence_negative_ratio}

    # 0: positive = negative
    # 1: positive > negative
    # 2: positive < negative
    def prediction_valence(self, tweets):
        valence = self.discretize_valence(tweets)
        valence_ratio = discretize_comparison(valence['valence_positive_ratio'], valence['valence_negative_ratio'])
        valence_amount = discretize_comparison(valence['valence_positive_amount'], valence['valence_negative_amount'])
        return {'valence_amount': valence_amount,
                'valence_ratio': valence_ratio}

    def predominant_emotion(self, tweets):
        predominant_emotion = predominant_category(tweets, self.emotions_categories)
        return {'predominant_emotion': predominant_emotion}

    def predominant_sentiment(self, tweets):
        predominant_sentiment = predominant_category(tweets, self.sentiment_categories)
        return {'predominant_sentiment': predominant_sentiment}

    def discretize_emotions(self, tweets):
        return discretize_categories(tweets, self.emotions_categories)

    def discretize_sentiments(self, tweets):
        return discretize_categories(tweets, self.sentiment_categories)
