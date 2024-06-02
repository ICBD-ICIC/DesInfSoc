import os
import pandas as pd

FOLDER_PATH = 'dataset/outputs'
RESULT_FILE = 'dataset/context2_ONLY-ACTION-SPREAD20_K3_H4_P12-MULTICLASS.csv'

file_list = os.listdir(FOLDER_PATH)

df_append = pd.DataFrame()

for file in file_list:
    df_temp = pd.read_csv(FOLDER_PATH + "/" + file, header=None)
    df_append = pd.concat([df_append, df_temp], ignore_index=True)
df_append.to_csv(RESULT_FILE, index=False)

print(len(df_append))
