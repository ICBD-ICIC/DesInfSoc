import pandas as pd

ORIGINAL_FILE = '../dataset/itrust/emotions.csv'
RESULT_FILE = '../outputs/emotions.csv'

original = pd.read_csv(ORIGINAL_FILE)
original = original.loc[:, ~original.columns.str.contains('^Unnamed')]  # remove previous index

original.to_csv(RESULT_FILE, index=False)
