# Calls Symanto Personality Traits API to get user's personality traits

import json
import time

import pandas as pd
import requests
import winsound

with open("config.json", "r") as config_file:
    config = json.load(config_file)

PERSONALITY_TRAITS_RESULTS = '../outputs/personality_traits/symanto_personality_traits_{0}.csv'.format(time.time())
USERS_TWEETS_FILE = '../dataset/users_min_10_all_tweets.csv'
SYMANTO_PERSONALITY_TRAITS_URL = "https://personality-traits.p.rapidapi.com/personality"


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


def get_symanto_personality_traits(users):
    headers = {
        "content-type": "application/json",
        "Accept": "application/json",
        "X-RapidAPI-Key": config["rapidAPI-Key"],
        "X-RapidAPI-Host": "personality-traits.p.rapidapi.com"
    }
    querystring = {"all": "true"}
    response = requests.post(SYMANTO_PERSONALITY_TRAITS_URL, json=payload(users), headers=headers, params=querystring)
    error_streak = 0
    sleep_time = 0

    while response.status_code != 200 and error_streak < 1:
        print(' ---> Error steak: {0}'.format(error_streak))
        print(' ---> Error {0} - {1}'.format(response.status_code, response.reason))
        error_streak += 1
        sleep_time += 1
        time.sleep(sleep_time)
        response = requests.post(SYMANTO_PERSONALITY_TRAITS_URL, json=payload(users), headers=headers, params=querystring)

    return response


user_messages = pd.read_csv(USERS_TWEETS_FILE)
results_rows = []
start = 0
steps = 32
end = start + steps

while len(user_messages[start:end] != 0):
    try:
        print('Processing user messages[{0}:{1}]'.format(start, end))
        results = get_symanto_personality_traits(user_messages[start:end])
        for result in results.json():
            predictions = result['predictions']
            results_rows.append({'id': result['id'],
                                 predictions[0]['prediction']: predictions[0]['probability'],
                                 predictions[1]['prediction']: predictions[1]['probability']})
        print('Done processing user messages[{0}:{1}]'.format(start, end))
        time.sleep(1)
    except Exception as e:
        print(e)
        print('Start: {0}'.format(start))
        print('End: {0}'.format(end))
        break
    start = end
    end += steps


# Save results
results = pd.DataFrame(results_rows)
results.to_csv(PERSONALITY_TRAITS_RESULTS, index=False)
print(user_messages)
winsound.Beep(440, 1000)
