import base_model
from sklearn.ensemble import RandomForestClassifier
import time
import json
from sklearn.model_selection import train_test_split

N_RUNS = 30

start = time.time()


BEST_ATTRIBUTES_SPREAD20 = {
    '28': {'n_estimators': 100, 'min_samples_leaf': 5},
    '30': {'n_estimators': 100, 'min_samples_leaf': 5},
    '31': {'n_estimators': 100, 'min_samples_leaf': 5},
    '32': {'n_estimators': 100, 'min_samples_leaf': 10},
    '34': {'n_estimators': 100, 'min_samples_leaf': 5},
    '36': {'n_estimators': 100, 'min_samples_leaf': 10}
}

BEST_ATTRIBUTES_ONLY_ACTION_SPREAD20 = {
    '28': {'n_estimators': 100 , 'min_samples_leaf': 5},
    '30': {'n_estimators': 100, 'min_samples_leaf': 5},
    '31': {'n_estimators': 100, 'min_samples_leaf': 5},
    '32': {'n_estimators': 100, 'min_samples_leaf': 10},
    '34': {'n_estimators': 100, 'min_samples_leaf': 10},
    '36': {'n_estimators': 100, 'min_samples_leaf': 5}
}

BEST_ATTRIBUTES_SPREAD60 = {
    '28': {'n_estimators': 100, 'min_samples_leaf': 10},
    '30': {'n_estimators': 100, 'min_samples_leaf': 10},
    '31': {'n_estimators': 100, 'min_samples_leaf': 5},
    '32': {'n_estimators': 100, 'min_samples_leaf': 10},
    '34': {'n_estimators': 100, 'min_samples_leaf': 10},
    '36': {'n_estimators': 100, 'min_samples_leaf': 10}
}

BEST_ATTRIBUTES_BALANCED = {
    'context_SPREAD60_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD60,
    'context_SPREAD20_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD20,
    'context_ONLY-ACTION-SPREAD20_K3_H4_P12-BINARY': BEST_ATTRIBUTES_ONLY_ACTION_SPREAD20
}

attributes = BEST_ATTRIBUTES_BALANCED[base_model.dataset_name][base_model.prediction]

X, y = base_model.get_dataset()

metrics = []

for i in range(N_RUNS):
    X, y = base_model.get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=base_model.TEST_SIZE, random_state=i)
    X_train, y_train = base_model.balance_train(X_train, y_train)
    model = RandomForestClassifier(n_estimators=attributes['n_estimators'], min_samples_leaf=attributes['min_samples_leaf'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.append(base_model.get_metrics(y_test, y_pred))
    print('{} done'.format(i))

with open(base_model.get_output_filepath('random_forest'), 'w') as file:
    file.write(json.dumps(metrics, indent=4))

print('FINISHED after {} seconds'.format(time.time() - start))
