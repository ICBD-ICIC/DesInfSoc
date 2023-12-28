import json
import time

import pandas as pd
import requests
import winsound

with open("config.json", "r") as config_file:
    config = json.load(config_file)

COMMUNICATION_STYLE_RESULTS = 'outputs/personality2/symanto_communication_style_{0}.csv'.format(time.time())
USERS_MESSAGES_FILE = 'dataset/network_users_tweets_to_analyze.csv'
SYMANTO_COMMUNICATION_STYLE_URL = "https://communication-style.p.rapidapi.com/communication"


def payload(users):
    payload = []
    for index, user in users.iterrows():
        user_payload = {
                "id": str(user['id']),
                "language": "en",
                "text": str(user['tweet_content'])
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
    response = requests.post(SYMANTO_COMMUNICATION_STYLE_URL, json=payload(users), headers=headers, params=querystring)
    error_streak = 0
    sleep_time = 0

    while response.status_code != 200 and error_streak < 1:
        print(' ---> Error steak: {0}'.format(error_streak))
        print(' ---> Error {0} - {1}'.format(response.status_code, response.reason))
        error_streak += 1
        sleep_time += 1
        time.sleep(sleep_time)
        response = requests.post(SYMANTO_COMMUNICATION_STYLE_URL, json=payload(users), headers=headers, params=querystring)

    return response


user_messages = pd.read_csv(USERS_MESSAGES_FILE)
results_rows = []
start = 25024
steps = 32
end = start + steps

while len(user_messages[start:end] != 0):
    try:
        print('Processing user messages[{0}:{1}]'.format(start, end))
        results = get_symanto_communication_style(user_messages[start:end])
        for result in results.json():
            predictions = result['predictions']
            results_rows.append({'id': result['id'],
                                 '{0}'.format(predictions[0]['prediction']): predictions[0]['probability'],
                                 '{0}'.format(predictions[1]['prediction']): predictions[1]['probability'],
                                 '{0}'.format(predictions[2]['prediction']): predictions[2]['probability'],
                                 '{0}'.format(predictions[3]['prediction']): predictions[3]['probability']})
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
