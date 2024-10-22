# Uses roberta to get emotions for all tweets in the context. No preprocessing is done

import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#CONTEXT_TWEETS_FILE = '../dataset/context_tweets.csv'
CONTEXT_TWEETS_FILE = '../dataset/emotion_too_long_original.csv'
EMOTIONS_OUTPUT_FILE = '../outputs/emotion_{0}.csv'.format(time.time())
ERRORS = '../outputs/emotion_too_long_{}.csv'.format(time.time())

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

all_tweets = pd.read_csv(CONTEXT_TWEETS_FILE)
all_tweets_emotions = []
errors = []

model_name = "j-hartmann/emotion-english-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

for index, row in all_tweets.iterrows():
    try:
        #inputs = tokenizer(row['text'], return_tensors="pt", padding=True)
        inputs = tokenizer(row['text'], return_tensors="pt", padding=True, truncation = True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).numpy()
        tweet_emotions = predicted_emotions = {EMOTIONS[i]: prob for i, prob in enumerate(probabilities[0])}
        tweet_emotions['id'] = row['id']

        all_tweets_emotions.append(tweet_emotions)
    except Exception as e:
        errors.append({'id': row['id'], 'text': row['text']})

emotions_df = pd.DataFrame(all_tweets_emotions)
emotions_df.to_csv(EMOTIONS_OUTPUT_FILE, index=False)

errors_df = pd.DataFrame(errors)
errors_df.to_csv(ERRORS, index=False)
