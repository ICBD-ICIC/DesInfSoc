import os
import pandas as pd

FOLDER_PATH = '../outputs/emotions'
RESULT_FILE = '../outputs/emotions/emotions.csv'

file_list = os.listdir(FOLDER_PATH)

df_append = pd.DataFrame()

for file in file_list:
            df_temp = pd.read_csv(FOLDER_PATH + "/" + file)
            df_append = pd.concat([df_append, df_temp], ignore_index=True)
df_append.to_csv(RESULT_FILE, index=False)
