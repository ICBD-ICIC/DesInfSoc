import base_model
from sklearn.linear_model import LogisticRegression
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

model = LogisticRegression(random_state=0, solver='saga')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('logistic_regression'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(model.get_params()))

print('FINISHED after {} seconds'.format(time.time()-start))
