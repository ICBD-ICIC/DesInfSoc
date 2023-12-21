import time
import pandas as pd
import requests
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

BIG_FIVE_RESULTS = 'outputs/personality/symanto_big_five_{0}.csv'.format(time.time())
USERS_MESSAGES_FILE = 'dataset/network_users_tweets_to_analyze.csv' # TODO: use network_users_tweets (renamed)
SYMANTO_BIG_FIVE_URL = "https://big-five-personality-insights.p.rapidapi.com/api/big5"


def get_symanto_big_five(user_data):
    payload = [
        {
            "id": str(user_data['id']),
            "language": "en", # We have mixed languages on the dataset.
            # TODO: check if we need to filter the non english parts of the tweets or what
            "text": str(user_data['tweet_content'])
        }
    ]
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": config["rapidAPI-Key"],
        "X-RapidAPI-Host": "big-five-personality-insights.p.rapidapi.com"
    }
    response = requests.post(SYMANTO_BIG_FIVE_URL, json=payload, headers=headers)
    return response


user_messages = pd.read_csv(USERS_MESSAGES_FILE)
successful_users_indexes = []
error_strikes = 0
big_five_results_rows = []

for index, user in user_messages.iterrows():
    try:
        result = get_symanto_big_five(user)
        big_five_results_rows.append(result.json()[0])
        successful_users_indexes.append(index)
    except Exception as e:
        print(e)
        error_strikes += 1
    if error_strikes > 2:
        break

# Save results
big_five_results = pd.DataFrame(big_five_results_rows)
big_five_results.to_csv(BIG_FIVE_RESULTS, index=False)

# Leave only the remaining users on the file
user_messages = user_messages.drop(index=successful_users_indexes)
user_messages.to_csv(USERS_MESSAGES_FILE, index=False)

