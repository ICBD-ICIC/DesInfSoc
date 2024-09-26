# Use this file to get all usernames and amounts from users with at least N tweets
# The output is a csv file located on OUTPUT_FILE

import pandas as pd

N = 10
OUTPUT_FILE = '../dataset/users_min_{}.csv'.format(N)

# Usernames and amounts from profiles.xlsl
data1 = pd.read_csv('../dataset/data1.csv', delimiter=';')
# Usernames and amounts from replies.xlsl
data2 = pd.read_csv('../dataset/data2.csv', delimiter=';')
# Usernames and amounts from influencers.xlsl
#data3 = pd.read_csv('dataset/data3.csv', delimiter=';')

df = pd.concat([data1, data2], ignore_index=True, sort=False).dropna()
all_users = df.groupby('user')['amount'].sum().reset_index().sort_values('amount')
all_users['amount'] = all_users['amount'].astype(int)

filtered_users = all_users[all_users['amount'] >= N].reset_index(drop=True)

filtered_users.to_csv(OUTPUT_FILE, index=False)

print(filtered_users)
