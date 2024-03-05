import pandas as pd
from ast import literal_eval

import datetime
from datetime import timedelta

# USERS = pd.read_csv('dataset/user-metrics-with-friends.csv').set_index('user_id')
# TWEETS = pd.read_csv('dataset/india-election-tweets-metrics.csv')
#
# print(USERS[~USERS.index.isin(TWEETS['user_id'])])
#
# # agarrar los friends que tienen tweets asi los puedo usar de index?
#
# uno = pd.read_csv('dataset/JOIN/itrust/abusive.csv').sort_values('id').reset_index(drop=True)
# dos = pd.read_csv('dataset/JOIN/itrust/emotions.csv').sort_values('id').reset_index(drop=True)
# tres = pd.read_csv('dataset/JOIN/itrust/mfd_binary.csv').sort_values('id').reset_index(drop=True)
# cuatro = pd.read_csv('dataset/JOIN/itrust/polarization.csv').sort_values('id').reset_index(drop=True)
# cinco = pd.read_csv('dataset/JOIN/itrust/sentiments.csv').sort_values('id').reset_index(drop=True)
# seis = pd.read_csv('dataset/JOIN/itrust/valence.csv').sort_values('id').reset_index(drop=True)
# original = pd.read_csv('dataset/JOIN/india-election-tweets-formatted-filtered-clean-final.csv').sort_values(
#     'id').reset_index()
# # print(uno.iloc[0]['id'])
# # print(uno.iloc[-1]['id'])
# # print(dos.iloc[0]['id'])
# # print(dos.iloc[-1]['id'])
# # print(tres.iloc[0]['id'])
# # print(tres.iloc[-1]['id'])
# # print(cuatro.iloc[0]['id'])
# # print(cuatro.iloc[-1]['id'])
# # print(cinco.iloc[0]['id'])
# # print(cinco.iloc[-1]['id'])
# # print(seis.iloc[0]['id'])
# # print(seis.iloc[-1]['id'])
# # print(original.iloc[0]['id'])
# # print(original.iloc[-1]['id'])
# #
# # # print(len(uno))
# # # print(len(uno.drop_duplicates()))
# # # print(len(dos))
# # # print(len(dos.drop_duplicates()))
# # # print(len(tres))
# # # print(len(tres.drop_duplicates()))
# # # print(len(cuatro))
# # # print(len(cuatro.drop_duplicates()))
# # # print(len(cinco))
# # # print(len(cinco.drop_duplicates()))
# # # print(len(seis))
# # # print(len(seis.drop_duplicates()))
# # # print(len(original))
# # # print(len(original.drop_duplicates()))
# # # 15013579
# # # 14999355
# # # 15013579
# # # 15000259
# # # 15013579
# # # 14999584
# # # 15013579
# # # 14999494
# # # 15013579
# # # 14999793
# # # 15013579
# # # 14999773
# # # 15013579
# # # 15013579
# # # print(original.index.difference(uno.index))
# # # print(original.index.difference(dos.index))
# # # print(original.index.difference(tres.index))
# # # print(original.index.difference(cuatro.index))
# # # print(original.index.difference(cinco.index))
# # # print(original.index.difference(seis.index))
# # # print()
# # # print(uno.index.difference(cuatro.index))
# #
# print(len(original))
# original = pd.concat([original, uno], axis=1)
# print(len(original))
# original = pd.concat([original, dos], axis=1)
# print(len(original))
# original = pd.concat([original, tres], axis=1)
# print(len(original))
# original = pd.concat([original, cuatro], axis=1)
# print(len(original))
# original = pd.concat([original, cinco], axis=1)
# print(len(original))
# original = pd.concat([original, seis], axis=1)
# original = original[
#     ['id', 'created_at', 'user_id', 'abusive_words_n', 'abusive_words_ratio', 'anger', 'disgust', 'fear',
#      'joy', 'neutral', 'sadness', 'surprise', 'vice_n', 'vice_ratio', 'virtue_n', 'virtue_ratio', 'polar_words_n',
#      'polar_words_ratio', 'sentiment-positive', 'sentiment-negative', 'sentiment-neutral', 'negative_words_n',
#      'negative_words_ratio', 'positive_words_n', 'positive_words_ratio']]

# original = pd.read_csv('dataset/india-election-tweets-metrics2.csv')
# original = original.drop(original.columns[1:7], axis=1)
# original['created_at'] = pd.to_datetime(original['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
# original = original.sort_values('created_at')
# print(original)
# original.to_csv('dataset/india-election-tweets-metrics.csv', index=False)

# USERS = pd.read_csv('dataset/user-metrics-with-friends.csv', converters={"friends": literal_eval})
# USERS = USERS.iloc[12000:13000].set_index('user_id') 928150009
# print(USERS.iloc[-1])

context = pd.read_csv('dataset/context_SPREAD60_K3_H4_P12.csv')
columns_to_replace = list(range(2, 6)) + list(range(7, 22)) + list(range(23, 26)) + list(range(27, 31)) + list(range(32, 36))

# Replace "4" with "3" in specified columns
context.iloc[:, columns_to_replace] = context.iloc[:, columns_to_replace].replace(4, 3)

context.to_csv('dataset/context_SPREAD60_K3_H4_P12-new.csv', index=False)

print(context['28'].value_counts())