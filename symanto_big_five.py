import json
import time

import pandas as pd
import requests
import winsound

with open("config.json", "r") as config_file:
    config = json.load(config_file)

BIG_FIVE_RESULTS = 'outputs/personality/symanto_big_five_{0}.csv'.format(time.time())
USERS_MESSAGES_FILE = 'dataset/network_users_tweets_to_analyze.csv'
SYMANTO_BIG_FIVE_URL = "https://big-five-personality-insights.p.rapidapi.com/api/big5"


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


def get_symanto_big_five(users):
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": config["rapidAPI-Key"],
        "X-RapidAPI-Host": "big-five-personality-insights.p.rapidapi.com"
    }

    response = requests.post(SYMANTO_BIG_FIVE_URL, json=payload(users), headers=headers)
    error_streak = 0
    sleep_time = 0

    while response.status_code != 200 and error_streak < 3:
        print(' ---> Error steak: {0}'.format(error_streak))
        print(' ---> Error {0} - {1}'.format(response.status_code, response.reason))
        error_streak += 1
        sleep_time += 1
        time.sleep(sleep_time)
        response = requests.post(SYMANTO_BIG_FIVE_URL, json=payload(users), headers=headers)

    return response


user_messages = pd.read_csv(USERS_MESSAGES_FILE)
big_five_results_rows = []
start = 142900
steps = 1000
end = start + steps

while len(user_messages[start:end] != 0):
    try:
        print('Processing user messages[{0}:{1}]'.format(start, end))
        results = get_symanto_big_five(user_messages[start:end])
        for result in results.json():
            big_five_results_rows.append(result)
        print('Done processing user messages[{0}:{1}]'.format(start, end))
    except Exception as e:
        print(e)
        print('Start: {0}'.format(start))
        print('End: {0}'.format(end))
        break
    start = end
    end += steps


# Save results
big_five_results = pd.DataFrame(big_five_results_rows)
big_five_results.to_csv(BIG_FIVE_RESULTS, index=False)
winsound.Beep(440, 1000)
