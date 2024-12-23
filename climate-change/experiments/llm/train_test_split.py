# Creates train and test files where the conversation ids do not appear in both

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "../../outputs/CONTEXT_LLM_pattern_matching_experiment#4.csv"

df = pd.read_csv(DATASET_PATH)

# Calculate sizes of each conversation_id
id_sizes = df['conversation_id'].value_counts()

# Perform a weighted split of the conversation_id values
train_ids, test_ids = train_test_split(
    id_sizes.index,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Ensure the split achieves close to the desired proportion
while True:
    train_df = df[df['conversation_id'].isin(train_ids)]
    test_df = df[df['conversation_id'].isin(test_ids)]

    test_ratio = len(test_df) / len(df)
    if 0.19 <= test_ratio <= 0.21:  # Tolerance for 20% split
        break

    # Adjust if necessary
    if test_ratio < 0.19:
        excess_id = np.random.choice(train_ids, 1)
        train_ids = [i for i in train_ids if i != excess_id]
        test_ids = np.append(test_ids, excess_id)
    elif test_ratio > 0.21:
        excess_id = np.random.choice(test_ids, 1)
        train_ids = np.append(train_ids, excess_id)
        test_ids = [i for i in test_ids if i != excess_id]

train_df.to_csv(DATASET_PATH.replace('.csv', '_train.csv'), index=False)
test_df.to_csv(DATASET_PATH.replace('.csv', '_test.csv'), index=False)
