import pandas as pd

original_path = 'dataset/context2_ONLY-ACTION-SPREAD20_K3_H4_P12-MULTICLASS.csv'
context = pd.read_csv(original_path)

# '33': 'abusive_amount_interval'
# '34': 'abusive_ratio_interval'
# '36': 'polarization_amount_interval'
# '37': 'polarization_ratio_interval'
# 0: [0,0.50)
# 1: [0.50,1]
context.iloc[:, [33, 34, 36, 37]] = context.iloc[:, [33, 34, 36, 37]].replace(1, 0)
context.iloc[:, [33, 34, 36, 37]] = context.iloc[:, [33, 34, 36, 37]].replace(2, 1)
context.iloc[:, [33, 34, 36, 37]] = context.iloc[:, [33, 34, 36, 37]].replace(3, 1)

# '39': 'predominant_emotion'
# 0: neutral, joy, surprise
# 1: fear, anger, disgust, sadness
context.iloc[:, 39] = context.iloc[:, 39].replace(4, 0)
context.iloc[:, 39] = context.iloc[:, 39].replace(6, 0)
context.iloc[:, 39] = context.iloc[:, 39].replace(3, 1)
context.iloc[:, 39] = context.iloc[:, 39].replace(2, 1)
context.iloc[:, 39] = context.iloc[:, 39].replace(5, 1)

# '40': 'mfd_ratio'
# '41': 'mfd_amount'
# 0: virtue >= vice
# 1: virtue < vice
context.iloc[:, [40, 41]] = context.iloc[:, [40, 41]].replace(1, 0)
context.iloc[:, [40, 41]] = context.iloc[:, [40, 41]].replace(2, 1)

# '44': 'valence_ratio'
# '45': 'valence_amount'
# 0: positive >= negative
# 1: positive < negative
context.iloc[:, [44, 45]] = context.iloc[:, [44, 45]].replace(1, 0)
context.iloc[:, [44, 45]] = context.iloc[:, [44, 45]].replace(2, 1)

# '48': 'predominant_sentiment'
# 0: positive, neutral
# 1: negative
context.iloc[:, 48] = context.iloc[:, 48].replace(1, 0)
context.iloc[:, 48] = context.iloc[:, 48].replace(2, 1)

context.to_csv(original_path.replace('MULTICLASS', 'BINARY'), index=False)
