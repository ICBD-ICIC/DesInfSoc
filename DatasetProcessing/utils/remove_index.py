import pandas as pd

ORIGINAL_FILE = '../dataset/india-election-tweets-metrics.csv'
RESULT_FILE = '../dataset/india-election-tweets-metrics.csv'

original = pd.read_csv(ORIGINAL_FILE)
original = original.loc[:, ~original.columns.str.contains('^Unnamed')]  # remove previous index

print(len(original))

original.to_csv(RESULT_FILE, index=False)
