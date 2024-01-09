import pandas as pd

personality = pd.read_csv('dataset/big_five.csv', dtype={'id': str})\
                .drop(['Source.Name'], axis=1)\
                .drop_duplicates(subset=['id'])\
                .dropna()\
                .reset_index(drop=True)
print(personality)


users_tweets = pd.read_csv('dataset/network_users_tweets_to_analyze.csv')
print(users_tweets)

personality.to_csv('dataset/big_five_clean.csv', index=False)
