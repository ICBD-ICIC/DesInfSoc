import pandas as pd

ORIGINAL_FILE = '../dataset/india-election-tweets-formatted-filtered-clean2.csv'
RESULT_FILE = '../dataset/india-election-tweets-formatted-filtered-clean.csv'

original = pd.read_csv(ORIGINAL_FILE).set_index('id')

print(len(original))

original = original.drop_duplicates()

print(len(original))

original.to_csv(RESULT_FILE)
