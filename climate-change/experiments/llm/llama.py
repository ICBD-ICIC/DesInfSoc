from ast import literal_eval
import random

import torch
from transformers import pipeline
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
discretization_path = os.path.join(current_dir, '..', '..', 'discretization')
sys.path.append(discretization_path)

from discretize_tweets_metrics import *  # Adjust based on the actual module content

# MAIN_USER = '@UserA' # TODO: map real usernames to synthetic ones
HIGH_LABEL = 'High'
LOW_LABEL = 'Low'
CONTEXT_AMOUNT = 10

experiment_type = 'pattern_matching'
#test_data = pd.read_csv("../../dataset/CONTEXT_LLM_{}-binary_test.csv".format(experiment_type))
test_data = pd.read_csv("../../outputs/CONTEXT_LLM_pattern_matching_PRUEBAS.csv",
                        converters={"user": literal_eval, "tweets": literal_eval})

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

discretizer = TweetsMetricsDiscretizer(experiment_type)

def context(tweets):
    return { **discretizer.discretize_abusive(tweets),
             **discretizer.discretize_polarization(tweets),
             **discretizer.predominant_emotion(tweets),
             **discretizer.discretize_emotions(tweets),
             **discretizer.discretize_mfd(tweets),
             **discretizer.discretize_valence(tweets),
             **discretizer.predominant_sentiment(tweets),
             **discretizer.discretize_sentiments(tweets)}


def personality_label(score):
    return HIGH_LABEL if score > 0.5 else LOW_LABEL


# Function to format the dictionary into a descriptive text
def format_personality_data(data, current_username):
    formatted_personality = f"@{current_username} has the following personality profile:\n\n"

    for key, value in data.items():
        if 'Value' in value.keys():
            score = float(value.get('Value'))
            label = personality_label(score)
            formatted_personality += f"{key}: ({label} - Score: {score:.2f}):\n"

            value = value.get('Traits')
        else:
            formatted_personality += f"{key}:\n"
        for value, score in value.items():
            score = float(score)
            formatted_personality += f"  - {value}: {score:.2f}\n"
        formatted_personality += "\n"

    return formatted_personality


def display_feature_label(feature):
    display_label = feature.split('ratio')[0]
    display_label = display_label.replace('_', ' ')
    display_label = display_label.replace('mfd', 'moral')
    return display_label

def format_tweet_features(features_averages):
    formatted_features = ""
    for key, value in features_averages.items():
        if "ratio" in key:
            formatted_features += (f"The amount of {display_feature_label(key)}words in the conversation "
                                   f"is {(HIGH_LABEL if value > 1 else LOW_LABEL).lower()}. ")
        elif key == "predominant_sentiment":
            formatted_features += (f"The predominant sentiment in the conversation "
                                   f"is {discretizer.sentiment_categories[value].replace('sentiment-', '')}. ")
        elif key == "predominant_emotion":
            formatted_features += (f"The predominant emotion in the conversation "
                                   f"is {discretizer.emotions_categories[value]}. ")
    return formatted_features

# Function to format the dictionary into a descriptive text
def format_tweets(tweets, current_username):
    formatted_tweets = f"@{current_username} has engaged in a Twitter conversation. Some tweets from that conversation are:\n\n"

    tweets_selection_ids = random.sample(tweets.keys(), CONTEXT_AMOUNT) # TODO: mantaint the first one?
    tweets_selection = {key: tweets[key] for key in tweets_selection_ids if key in tweets}

    for key, value in tweets_selection.items():
        formatted_tweets += f"  - {value['text']}\n"

    tweet_features = pd.DataFrame(tweets_selection).T
    features_averages = context(tweet_features)
    formatted_tweets += f"\n{format_tweet_features(features_averages)}\n"

    return formatted_tweets


personality_data = test_data.iloc[0]['user']
current_user = test_data.iloc[0]['current_username']
tweets = test_data.iloc[0]['tweets']

prompt = (f"{format_personality_data(personality_data, current_user)}"
          f"\n"
          f"{format_tweets(tweets, current_user)}"
          f"\n\n"
          f"Do you think @{current_user}'s response to the conversation "
          f"will have a {HIGH_LABEL.lower()} or {LOW_LABEL.lower()} amount of abusive words?")

print(prompt)

outputs = pipe(
    prompt,
    max_new_tokens = 10,
    do_sample=False,
    return_full_text = False
)
print(outputs[0]["generated_text"])

print(test_data.iloc[0]['abusive_ratio_interval_gt'])
print(test_data.iloc[0]['abusive_amount_interval_gt'])
