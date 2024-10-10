# Calls Symanto Communication Style API to get user's communication style

import json
import time

import pandas as pd
import requests
import winsound

with open("config.json", "r") as config_file:
    config = json.load(config_file)

COMMUNICATION_STYLE_RESULTS = '../outputs/communication_style/symanto_communication_style_{0}.csv'.format(time.time())
USERS_TWEETS_FILE = '../dataset/users_min_10_all_tweets.csv'
SYMANTO_COMMUNICATION_STYLE_URL = "https://communication-style.p.rapidapi.com/communication"


def payload(users):
    payload = []
    for index, user in users.iterrows():
        user_payload = {
                "id": str(user['user']),
                "language": "en",
                "text": str(user['text'])
            }
        payload.append(user_payload)
    return payload


def get_symanto_communication_style(users):
    headers = {
        "content-type": "application/json",
        "Accept": "application/json",
        "X-RapidAPI-Key": config["rapidAPI-Key"],
        "X-RapidAPI-Host": "communication-style.p.rapidapi.com"
    }
    querystring = {"all": "true"}
    time.sleep(10)
    response = requests.post(SYMANTO_COMMUNICATION_STYLE_URL, json=payload(users), headers=headers, params=querystring)
    error_streak = 0
    sleep_time = 30

    while response.status_code != 200 and error_streak < 1:
        print(' ---> Error steak: {0}'.format(error_streak))
        print(' ---> Error {0} - {1}'.format(response.status_code, response.reason))
        error_streak += 1
        sleep_time += 1
        time.sleep(sleep_time)
        response = requests.post(SYMANTO_COMMUNICATION_STYLE_URL, json=payload(users), headers=headers, params=querystring)
        print(response.status_code)
    return response


user_messages = pd.read_csv(USERS_TWEETS_FILE)
results_rows = []
start = 2078
steps = 10
end = start + steps

while len(user_messages[start:end]) != 0:
    try:
        print('Processing user messages[{0}:{1}]'.format(start, end))
        results = get_symanto_communication_style(user_messages[start:end])
        for result in results.json():
            predictions = result['predictions']
            results_rows.append({'id': result['id'],
                                 predictions[0]['prediction']: predictions[0]['probability'],
                                 predictions[1]['prediction']: predictions[1]['probability'],
                                 predictions[2]['prediction']: predictions[2]['probability'],
                                 predictions[3]['prediction']: predictions[3]['probability']})
        print('Done processing user messages[{0}:{1}]'.format(start, end))
    except Exception as e:
        print(e)
        print('Start: {0}'.format(start))
        print('End: {0}'.format(end))
        break
    start = end
    end += steps


# Save results
results = pd.DataFrame(results_rows)
results.to_csv(COMMUNICATION_STYLE_RESULTS, index=False)
winsound.Beep(440, 1000)
