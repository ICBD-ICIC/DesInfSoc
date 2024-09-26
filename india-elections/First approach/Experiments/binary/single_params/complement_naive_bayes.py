import base_model
from sklearn.naive_bayes import ComplementNB
import time
import json
import sys


def bool(string):
    if string.lower() == 'true':
        return True
    else:
        return False


start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

alpha = float(sys.argv[3])
norm = bool(sys.argv[4])

model = ComplementNB(alpha=alpha, norm=norm, force_alpha=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

model_params = '(alpha={}-norm={})'.format(alpha, norm)

with open(base_model.get_output_filepath('complement_naive_bayes', model_params), 'w') as file:
    file.write(json.dumps(metrics, indent=4))
    file.write(str(model.get_params()))

print('FINISHED after {} seconds'.format(time.time() - start))
