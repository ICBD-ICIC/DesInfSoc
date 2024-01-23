import pandas as pd

CSV_FILE = '../outputs/india-election-tweets-formatted-missing.csv'

df = pd.read_csv(CSV_FILE)
df.to_csv(CSV_FILE, index=False)
print(len(df))
