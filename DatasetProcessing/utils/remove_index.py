import pandas as pd

ORIGINAL_FILE = '../outputs/sentiments2.csv'
RESULT_FILE = '../outputs/sentiments.csv'

original = pd.read_csv(ORIGINAL_FILE)
original = original.loc[:, ~original.columns.str.contains('^Unnamed')]  # remove previous index

print(len(original))

original.to_csv(RESULT_FILE, index=False)
