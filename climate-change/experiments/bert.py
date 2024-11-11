# Based on https://www.kaggle.com/code/mexwell/bert-for-binary-classification

import pandas as pd
import time

start = time.time()

dataset = pd.read_csv('dataset/llm_dataset.csv')

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

X = dataset[CONTEXT_FEATURES]
y = dataset[prediction]
return X, y

