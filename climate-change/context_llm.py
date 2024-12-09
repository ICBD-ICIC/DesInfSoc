# Creates the dataset as needed for the LLM classifiers

# TODO: decide how many tweets we want to use per context and save only that amount and the calculated average
# I can also save the formated user features

from ast import literal_eval

import numpy as np
import pandas as pd
import time
from discretization.discretize_tweets_metrics import *

MAX_ROWS = 50
MAX_TWEETS = 10

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

experiment_type = 'pattern_matching'

USERS_FEATURES = pd.read_csv('dataset/for_llm_context/users_features_llm.csv')
tweets_ids_all = pd.read_csv('dataset/for_llm_context/input_and_ground_truth_llm.csv',
                         converters={"previous_posts_ids": literal_eval})
TWEETS_IDS = tweets_ids_all.sample(MAX_ROWS) # For testing purposes, remove later
CONTEXT_TWEETS = pd.read_csv('dataset/for_llm_context/context_tweets_llm.csv')
tweets_features = pd.read_csv('dataset/for_context/tweets_features_{}.csv'.format(experiment_type))
TWEETS_FEATURES = tweets_features.loc[:, ~tweets_features.columns.str.contains('^Unnamed')] #Drop previous index
TWEETS_FEATURES = TWEETS_FEATURES.replace({np.nan: None})

OUTPUT_FILE = 'outputs/CONTEXT_LLM_{}_{}.csv'.format(experiment_type, time.time())

discretizer = TweetsMetricsDiscretizer(experiment_type)

def format_user_features(user):
    formatted_user_features = {
        "Openness": {
            "Value": user['personality@openness'],
            "Traits": {
                "Adventurous": user['personality@adventurous'],
                "Artistic": user['personality@artistic'],
                "Emotionally aware": user['personality@emotionally_aware'],
                "Imaginative": user['personality@imaginative'],
                "Intellectual": user['personality@intellectual'],
                "Authority challenging": user['personality@authority_challenging']}},
        "Conscientiousness": {
            "Value": user['personality@conscientiousness'],
            "Traits": {
                "Disciplined": user['personality@disciplined'],
                "Dutiful": user['personality@dutiful'],
                "Cautious": user['personality@cautious'],
                "Achievement striving": user['personality@achievement_striving'],
                "Orderliness": user['personality@orderliness'],
                "Self Efficacy": user['personality@self_efficacy']
                }},
        "Extraversion": {
            "Value": user['personality@extraversion'],
            "Traits": {
                "Assertive": user['personality@assertive'],
                "Cheerful": user['personality@cheerful'],
                "Gregariousness": user['personality@gregariousness'],
                "Active": user['personality@active'],
                "Excitement seeking": user['personality@excitement_seeking'],
                "Outgoing": user['personality@outgoing']
            }},
        "Agreeableness": {
            "Value": user['personality@agreeableness'],
            "Traits": {
                "Altruism": user['personality@altruism'],
                "Cooperative": user['personality@cooperative'],
                "Modesty": user['personality@modesty'],
                "Trusting": user['personality@trusting'],
                "Sympathy": user['personality@sympathy'],
                "Uncompromising": user['personality@uncompromising']
                }},
        "Neuroticism": {
            "Value": user['personality@neuroticism'],
            "Traits": {
                "Melancholy": user['personality@melancholy'],
                "Stress prone": user['personality@stress_prone'],
                "Self conscious": user['personality@self_conscious'],
                "Immoderation": user['personality@immoderation'],
                "Fiery": user['personality@fiery'],
                "Prone to worry": user['personality@prone_to_worry']
                }},
        "Communication style": {
            "Information seeking": user['communication_style@information-seeking'],
            "Action seeking": user['communication_style@action-seeking'],
            "Fact oriented": user['communication_style@fact-oriented'],
            "Self revealing": user['communication_style@self-revealing']
        },
        "Personality traits": {
            "Rational": user['personality_traits@rational'],
            "Emotional": user['personality_traits@emotional']
        }
    }
    return formatted_user_features

def context(tweets):
    return { **discretizer.discretize_abusive(tweets),
             **discretizer.discretize_polarization(tweets),
             **discretizer.predominant_emotion(tweets),
             **discretizer.discretize_emotions(tweets),
             **discretizer.discretize_mfd(tweets),
             **discretizer.discretize_valence(tweets),
             **discretizer.predominant_sentiment(tweets),
             **discretizer.discretize_sentiments(tweets)}

#Dessigned for multiple tweets but in our case we have only 1
def ground_truth(tweets):
    all_ground_truths = { **discretizer.discretize_abusive(tweets),
                         **discretizer.discretize_polarization(tweets),
                         **discretizer.predominant_emotion(tweets),
                         **discretizer.prediction_mfd(tweets),
                         **discretizer.prediction_valence(tweets),
                         **discretizer.predominant_sentiment(tweets) }
    return { str(key) + '_gt': val for key, val in all_ground_truths.items()}

context_rows = []

for index, row in TWEETS_IDS.iterrows():
    user_features = USERS_FEATURES[USERS_FEATURES['id'] == row['username']].iloc[0].drop('id').to_dict()
    prediction_tweets = TWEETS_FEATURES[TWEETS_FEATURES['id'] == row['user_reply_id']]

    context_tweets = CONTEXT_TWEETS[CONTEXT_TWEETS['id'].isin(row['previous_posts_ids'])]
    context_tweets = context_tweets.sort_values(by='created_at', ascending=True)
    context_sample = context_tweets[0:min(10, len(context_tweets))]
    tweets_features = TWEETS_FEATURES[TWEETS_FEATURES['id'].isin(row['previous_posts_ids'])]

    context_rows.append({ 'conversation_id': row['conversation_id'],
                          'current_username': row['username'],
                          'user': format_user_features(user_features),
                          'tweets_sample': list(context_sample.sort_values(by='created_at', ascending=True)['text']),
                          'context_features': context(tweets_features),
                          **ground_truth(prediction_tweets)})
context_df = pd.DataFrame(context_rows)
context_df.to_csv(OUTPUT_FILE, index=False)
