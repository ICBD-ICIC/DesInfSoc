import base_model
from sklearn.naive_bayes import ComplementNB
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

BEST_ATTRIBUTES = {
    '28': {'norm': False, 'alpha': 0.2},
    '30': {'norm': False, 'alpha': 0.2},
    '31': {'norm': False, 'alpha': 0.2},
    '32': {'norm': False, 'alpha': 0.2},
    '34': {'norm': False, 'alpha': 0.1},
    '36': {'norm': False, 'alpha': 0.2}
}

attributes = BEST_ATTRIBUTES[base_model.prediction]

model = ComplementNB(force_alpha=True, alpha=attributes['alpha'], norm=attributes['norm'])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('complement_naive_bayes'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(model.get_params()))

print('FINISHED after {} seconds'.format(time.time()-start))

