import json
import time

import pandas as pd
import requests
import winsound

with open("config.json", "r") as config_file:
    config = json.load(config_file)

EKMAN_EMOTIONS_RESULTS = 'outputs/emotions/symanto_ekman_emotions_{0}.csv'.format(time.time())
DATASET = 'dataset/india-election-tweets-formatted-filtered-clean.csv'
SYMANTO_EKMAN_EMOTIONS_URL = "https://ekman-emotion-analysis.p.rapidapi.com/ekman-emotion"


def payload(tweets):
    payload = []
    for index, tweet in tweets.iterrows():
        tweet_payload = {
                "id": str(tweet['id']),
                "language": "en",
                "text": str(tweet['text'])
            }
        payload.append(tweet_payload)
    return payload


def get_symanto_ekman_emotions(tweets):
    headers = {
        "content-type": "application/json",
        "Accept": "application/json",
        "X-RapidAPI-Key": config["rapidAPI-Key"],
        "X-RapidAPI-Host": "ekman-emotion-analysis.p.rapidapi.com"
    }
    querystring = {"all": "true"}
    response = requests.post(SYMANTO_EKMAN_EMOTIONS_URL, json=payload(tweets), headers=headers, params=querystring)
    error_streak = 0
    sleep_time = 0

    while response.status_code != 200 and error_streak < 1:
        print(' ---> Error steak: {0}'.format(error_streak))
        print(' ---> Error {0} - {1}'.format(response.status_code, response.reason))
        error_streak += 1
        sleep_time += 1
        time.sleep(sleep_time)
        response = requests.post(SYMANTO_EKMAN_EMOTIONS_URL, json=payload(tweets), headers=headers, params=querystring)
    return response


tweets = pd.read_csv(DATASET)
results_rows = []
start = 8040
steps = 8
end = start + steps

while len(tweets[start:end] != 0):
    try:
        print('Processing tweets[{0}:{1}]'.format(start, end))
        results = get_symanto_ekman_emotions(tweets[start:end])
        for result in results.json():
            predictions = result['predictions']
            results_rows.append({'id': result['id'],
                                 '{0}'.format(predictions[0]['prediction']): predictions[0]['probability'],
                                 '{0}'.format(predictions[1]['prediction']): predictions[1]['probability'],
                                 '{0}'.format(predictions[2]['prediction']): predictions[2]['probability'],
                                 '{0}'.format(predictions[3]['prediction']): predictions[3]['probability'],
                                 '{0}'.format(predictions[4]['prediction']): predictions[4]['probability'],
                                 '{0}'.format(predictions[5]['prediction']): predictions[5]['probability'],
                                 '{0}'.format(predictions[6]['prediction']): predictions[6]['probability']})
        print('Done processing tweets[{0}:{1}]'.format(start, end))
    except Exception as e:
        print(e)
        print('Start: {0}'.format(start))
        print('End: {0}'.format(end))
        break
    start = end
    end += steps


# Save results
results = pd.DataFrame(results_rows)
results.to_csv(EKMAN_EMOTIONS_RESULTS, index=False)
winsound.Beep(440, 1000)
