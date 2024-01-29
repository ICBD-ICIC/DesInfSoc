import os
import pandas as pd

FOLDER_PATH = '../outputs/tweets'
RESULT_FILE = '../dataset/india-election-tweets-formatted-filtered-clean-final.csv'

file_list = os.listdir(FOLDER_PATH)

df_append = pd.DataFrame()

for file in file_list:
            df_temp = pd.read_csv(FOLDER_PATH + "/" + file)
            df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]  # remove previous index
            df_append = pd.concat([df_append, df_temp], ignore_index=True)
df_append.to_csv(RESULT_FILE, index=False)