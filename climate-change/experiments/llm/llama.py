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

EXPERIMENT_NUMBER = 5
EXPERIMENT_TYPE = 'pattern_matching'
FEATURE_TYPE = 'interval' #categorical, interval, interval_comparison
FEATURE = 'abusive'

WITH_EXAMPLES = False

OUTPUT_FILE = f'../../outputs/experiment#{EXPERIMENT_NUMBER}-{FEATURE}.csv'

HIGH_LABEL = 'High'
LOW_LABEL = 'Low'

MAX_TWEETS = 5

DATASET_PATH = '../../outputs/experiments/experiment#{}/CONTEXT_LLM_pattern_matching_experiment#{}.csv'.format(EXPERIMENT_NUMBER, EXPERIMENT_NUMBER)

test_data = pd.read_csv(DATASET_PATH.replace('.csv', '_test.csv'),
                        converters={"user": literal_eval,
                                    "tweets_sample": literal_eval,
                                    "context_features": literal_eval})[0:5]
examples_data = pd.read_csv(DATASET_PATH.replace('.csv', '_train.csv'),
                        converters={"user": literal_eval,
                                    "tweets_sample": literal_eval,
                                    "context_features": literal_eval})

if FEATURE_TYPE == 'interval':
    positive_examples = examples_data[
        (examples_data[f'{FEATURE}_amount_interval_gt'] > 1) |
        (examples_data[f'{FEATURE}_ratio_interval_gt'] > 1)
    ]
    negative_examples = examples_data[
        (examples_data[f'{FEATURE}_amount_interval_gt'] < 2) &
        (examples_data[f'{FEATURE}_ratio_interval_gt'] < 2)
    ]

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

discretizer = TweetsMetricsDiscretizer(EXPERIMENT_TYPE)

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

def prediction_task(current_user):
    if FEATURE_TYPE == 'interval':
        return (f"In one word, predict if @{current_user}'s response to the conversation will have a {HIGH_LABEL} or "
                f"{LOW_LABEL} amount of {FEATURE} words.")

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
        prompt += f"{ground_truth(data) + '.'}"
    return prompt

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['row_index', 'prompt', 'result', 'ground_truth'])

    for index, row in test_data.iterrows():
        positive_example = positive_examples.sample(1).iloc[0]
        negative_example = negative_examples.sample(1).iloc[0]

        main_prompt = "<|begin_of_text|>"

        if WITH_EXAMPLES:
            main_prompt += (f"{get_single_prompt(positive_example, True)}\n"
                       f"{get_single_prompt(negative_example, True)}\n")

        main_prompt += f"{get_single_prompt(row, False)}\n"

        print(main_prompt)

        outputs = pipe(
            main_prompt,
            max_new_tokens=10,
            do_sample=False,
            return_full_text=False
        )
        response = outputs[0]["generated_text"]
        csv_writer.writerow([index, main_prompt, response, ground_truth(row)])
