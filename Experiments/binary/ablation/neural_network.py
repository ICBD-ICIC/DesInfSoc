import base_model
from sklearn.neural_network import MLPClassifier
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

BEST_ATTRIBUTES_SPREAD20 = {
    '28': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '30': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '31': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '32': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '34': {'hidden_layer_sizes': (15,), 'batch_size': 200},
    '36': {'hidden_layer_sizes': (30,), 'batch_size': 100}
}

BEST_ATTRIBUTES_SPREAD60 = {
    '28': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '30': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '31': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '32': {'hidden_layer_sizes': (20,), 'batch_size': 200},
    '34': {'hidden_layer_sizes': (15,), 'batch_size': 200},
    '36': {'hidden_layer_sizes': (30,), 'batch_size': 100}
}

BEST_ATTRIBUTES = {
    'context_SPREAD60_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD60,
    'context_SPREAD20_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD20
}

attributes = BEST_ATTRIBUTES[base_model.dataset_name][base_model.prediction]

model = MLPClassifier(hidden_layer_sizes=attributes['hidden_layer_sizes'], batch_size=attributes['batch_size'])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('neural_network'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(model.get_params()))

print('FINISHED after {} seconds'.format(time.time()-start))

