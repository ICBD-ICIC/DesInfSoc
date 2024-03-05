import pandas as pd

context = pd.read_csv('dataset/context_SPREAD20_K3_H4_P12.csv')

# '27': 'abusive_amount_interval'
# '28': 'abusive_ratio_interval'
# '29': 'polarization_amount_interval'
# '30': 'polarization_ratio_interval'
# 0: [0,0.50)
# 1: [0.50,1]
context.iloc[:, [27, 28, 29, 30]] = context.iloc[:, [27, 28, 29, 30]].replace(1, 0)
context.iloc[:, [27, 28, 29, 30]] = context.iloc[:, [27, 28, 29, 30]].replace(2, 1)
context.iloc[:, [27, 28, 29, 30]] = context.iloc[:, [27, 28, 29, 30]].replace(3, 1)

# '31': 'predominant_emotion'
# 0: neutral, joy, surprise
# 1: fear, anger, disgust, sadness
context.iloc[:, 31] = context.iloc[:, 31].replace(4, 0)
context.iloc[:, 31] = context.iloc[:, 31].replace(6, 0)
context.iloc[:, 31] = context.iloc[:, 31].replace(3, 1)
context.iloc[:, 31] = context.iloc[:, 31].replace(2, 1)
context.iloc[:, 31] = context.iloc[:, 31].replace(5, 1)

# '32': 'mfd_ratio'
# '33': 'mfd_amount'
# 0: virtue >= vice
# 1: virtue < vice
context.iloc[:, [32, 33]] = context.iloc[:, [32, 33]].replace(1, 0)
context.iloc[:, [32, 33]] = context.iloc[:, [32, 33]].replace(2, 1)

# '34': 'valence_ratio'
# '35': 'valence_amount'
# 0: positive >= negative
# 1: positive < negative
context.iloc[:, [34, 35]] = context.iloc[:, [34, 35]].replace(1, 0)
context.iloc[:, [34, 35]] = context.iloc[:, [34, 35]].replace(2, 1)

# '36': 'predominant_sentiment'
# 0: positive, neutral
# 1: negative
context.iloc[:, 36] = context.iloc[:, 36].replace(1, 0)
context.iloc[:, 36] = context.iloc[:, 36].replace(2, 1)

context.to_csv('dataset/context_SPREAD20_K3_H4_P12-BINARY.csv', index=False)

