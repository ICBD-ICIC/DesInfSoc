import csv
import math
import time
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

EXPERIMENT_NUMBER = 5.1
EXPERIMENT_TYPE = 'pattern_matching'
FEATURE_TYPE = 'categorical' #categorical, interval, interval_comparison
FEATURE = 'emotion'

WITH_EXAMPLES = True

EXPERIMENT_PATH = f'../../outputs/experiments/experiment#{math.floor(EXPERIMENT_NUMBER)}/'
DATASET_PATH = f'{EXPERIMENT_PATH}CONTEXT_LLM_pattern_matching_experiment#{math.floor(EXPERIMENT_NUMBER)}.csv'
OUTPUT_FILE = f'{EXPERIMENT_PATH}/experiment#{EXPERIMENT_NUMBER}-{FEATURE}-{"few_shot" if WITH_EXAMPLES else "zero_shot"}.csv'

HIGH_LABEL = 'High'
LOW_LABEL = 'Low'

MAX_TWEETS = 5

test_data = pd.read_csv(DATASET_PATH.replace('.csv', '_test.csv'),
                        converters={"user": literal_eval,
                                    "tweets_sample": literal_eval,
                                    "context_features": literal_eval})[0:5]
examples_data = pd.read_csv(DATASET_PATH.replace('.csv', '_train.csv'),
                        converters={"user": literal_eval,
                                    "tweets_sample": literal_eval,
                                    "context_features": literal_eval})

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

discretizer = TweetsMetricsDiscretizer(EXPERIMENT_TYPE)

def get_examples():
    if FEATURE_TYPE == 'interval':
        positive_mask = (examples_data[f'{FEATURE}_amount_interval_gt'] > 1) | \
                        (examples_data[f'{FEATURE}_ratio_interval_gt'] > 1)
        negative_mask = (examples_data[f'{FEATURE}_amount_interval_gt'] < 2) & \
                        (examples_data[f'{FEATURE}_ratio_interval_gt'] < 2)

        positive_example = examples_data[positive_mask].sample(1).iloc[0]
        negative_example = examples_data[negative_mask].sample(1).iloc[0]

        return [positive_example, negative_example]
    elif FEATURE_TYPE == 'categorical':
        if FEATURE == 'sentiment':
            categories = discretizer.sentiment_categories
        elif FEATURE == 'emotion':
            categories = discretizer.emotions_categories
        categories = range(0, len(categories))
        feature = f'predominant_{FEATURE}_gt'
        categories_examples = []
        for category in categories:
            categories_examples.append(examples_data[examples_data[feature] == category].sample(1).iloc[0])
        return categories_examples

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
    formatted_tweets = f"@{current_username} has engaged in a Twitter conversation. The last tweets from that conversation are:\n"

    for tweet in tweets[0:min(MAX_TWEETS, len(tweets))]:
        formatted_tweets += f"  - \"{tweet}\"\n"

    formatted_tweets += f"\n{format_tweet_features(context_features)}"

    return formatted_tweets

def ground_truth(data):
    if FEATURE_TYPE == 'interval':
        if (data[f'{FEATURE}_amount_interval_gt'] > 1) or (data[f'{FEATURE}_ratio_interval_gt'] > 1):
            return HIGH_LABEL
        else:
            return LOW_LABEL
    if FEATURE_TYPE == 'categorical':
        if FEATURE == 'sentiment':
            return discretizer.sentiment_categories[data['predominant_sentiment_gt']].replace('sentiment-', '').capitalize()
        elif FEATURE == 'emotion':
            return discretizer.emotions_categories[data['predominant_emotion_gt']].capitalize()

def prediction_task(current_user):
    if FEATURE_TYPE == 'interval':
        return (f"In one word, predict if @{current_user}'s response to the conversation will have a {HIGH_LABEL} or "
                f"{LOW_LABEL} amount of {FEATURE} words. Your answer should be {HIGH_LABEL} or {LOW_LABEL}.")
    if FEATURE_TYPE == 'categorical':
        if FEATURE == 'sentiment':
            categories = [item.replace('sentiment-', '').capitalize() for item in discretizer.sentiment_categories]
        elif FEATURE == 'emotion':
            categories = [item.capitalize() for item in discretizer.emotions_categories]
        categories = ', '.join(categories[:-1]) + ', or ' + categories[-1]
        return (f"In one word, predict which {FEATURE} will have @{current_user}'s response to the conversation."
                f" Your answer should be {categories}.")

def get_single_prompt(data, is_example):
    personality_data = data['user']
    current_user = data['current_username']
    context_features = data['context_features']
    tweets = data['tweets_sample']
    prompt = (f"<|start_header_id|>system<|end_header_id|>{format_personality_data(personality_data, current_user)}\n"
              f"{format_tweets(tweets, current_user, context_features)}<|eot_id|>"
              f"<|start_header_id|>user<|end_header_id|>{prediction_task(current_user)}<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>")
    if is_example:
        prompt += f"{ground_truth(data) + '<|eot_id|>'}"
    return prompt

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['row_index', 'prompt', 'result', 'ground_truth'])

    for index, row in test_data.iterrows():
        start = time.time()

        main_prompt = "<|begin_of_text|>"

        if WITH_EXAMPLES:
            examples = get_examples()
            for example in examples:
                main_prompt += f"{get_single_prompt(example, True)}\n"

        main_prompt += f"{get_single_prompt(row, False)}\n"

        outputs = pipe(
            main_prompt,
            max_new_tokens=10,
            do_sample=False,
            return_full_text=False
        )
        response = outputs[0]["generated_text"]
        csv_writer.writerow([index, main_prompt, response, ground_truth(row)])

        elapsed = time.time() - start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")