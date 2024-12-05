# experiment1

import csv
from ast import literal_eval
import torch
from transformers import pipeline
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
discretization_path = os.path.join(current_dir, '..', '..', 'discretization')
sys.path.append(discretization_path)

from discretize_tweets_metrics import *

OUTPUT_FILE = '../../outputs/experiment#1-abusive.csv'

HIGH_LABEL = 'High'
LOW_LABEL = 'Low'

experiment_type = 'pattern_matching'
test_data = pd.read_csv("../../outputs/CONTEXT_LLM_pattern_matching_experiment#1.csv",
                        converters={"user": literal_eval, "tweets_sample": literal_eval, "context_features": literal_eval})

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

discretizer = TweetsMetricsDiscretizer(experiment_type)

def personality_label(score):
    return HIGH_LABEL if score > 0.5 else LOW_LABEL


def format_personality_data(data, current_username):
    formatted_personality = f"@{current_username} has the following personality profile:\n"

    for key, value in data.items():
        if 'Value' in value.keys():
            score = float(value.get('Value'))
            label = personality_label(score)
            formatted_personality += f"  - {key}: {label} (Score: {score:.2f})\n"
        elif key == "Personality traits":
            formatted_personality += f"  - {key}: {'Rational' if value['Rational'] > 0.5 else 'Emotional'}\n"
        else:
            formatted_communication_style = ', '.join(key for key, value in value.items() if value > 0.5)
            formatted_personality += f"  - {key}: {formatted_communication_style}\n"
    return formatted_personality


def display_feature_label(feature):
    display_label = feature.split('ratio')[0]
    display_label = display_label.replace('_', ' ')
    display_label = display_label.replace('mfd', 'moral').strip()
    return display_label.capitalize()

def format_tweet_features(features_averages):
    formatted_features = "Conversation metrics:\n"
    for key, value in features_averages.items():
        if "ratio" in key:
            amount_value = features_averages[key.replace('ratio', 'amount')]
            feature_value = (HIGH_LABEL if ((value > 1) or (amount_value > 1)) else LOW_LABEL)
            formatted_features += f"  - {display_feature_label(key)} words: {feature_value}\n"
        elif key == "predominant_sentiment":
            formatted_features += (f"  - Predominant sentiment: "
                                   f"{discretizer.sentiment_categories[value].replace('sentiment-', '').capitalize()}\n")
        elif key == "predominant_emotion":
            formatted_features += (f"  - Predominant emotion: "
                                   f"{discretizer.emotions_categories[value].capitalize()}\n")
    return formatted_features

def format_tweets(tweets, current_username, context_features):
    formatted_tweets = f"@{current_username} has engaged in a Twitter conversation. Some tweets from that conversation are:\n"

    for tweet in tweets:
        formatted_tweets += f"  - {tweet}\n"

    formatted_tweets += f"\n{format_tweet_features(context_features)}\n"

    return formatted_tweets

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['row_index', 'tweets_amount', 'prompt', 'result', 'ground_truth'])

    for index, row in test_data.iterrows():
        personality_data = row['user']
        current_user = row['current_username']
        context_features = row['context_features']
        ground_truth = (HIGH_LABEL if ((row['abusive_amount_interval_gt'] > 1) or (row['abusive_ratio_interval_gt'] > 1)) else LOW_LABEL)

        for amount in range(2, 11, 2):
            tweets = row['tweets_sample'][0:amount]

            prompt = (f"In one word, predict if @{current_user}'s response to the conversation "
                      f"will have a {HIGH_LABEL} or {LOW_LABEL} amount of abusive words."
                      f"{format_personality_data(personality_data, current_user)}"
                      f"\n"
                      f"{format_tweets(tweets, current_user, context_features)}"
                      f"\n"
                      f"In one word, predict if @{current_user}'s response to the conversation "
                      f"will have a {HIGH_LABEL} or {LOW_LABEL} amount of abusive words.")

            outputs = pipe(
                prompt,
                max_new_tokens=50,
                do_sample=False,
                return_full_text=False
            )
            response = outputs[0]["generated_text"]

            csv_writer.writerow([index, amount, prompt, response, ground_truth])
