import pandas as pd

data1 = pd.read_csv('dataset/data1.csv', delimiter=';')
data2 = pd.read_csv('dataset/data2.csv', delimiter=';')
data3 = pd.read_csv('dataset/data3.csv', delimiter=';')

df = pd.concat([data1, data2, data3], ignore_index=True, sort=False).dropna()
result = df.groupby('user')['amount'].sum().reset_index().sort_values('amount')
result['amount'] = result['amount'].astype(int)

result.to_csv('all.csv', index=False)

print(result)