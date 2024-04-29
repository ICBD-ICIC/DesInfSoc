import base_model
from sklearn.ensemble import RandomForestClassifier
import time
import json
from sklearn.model_selection import KFold

K = 10

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
kf = KFold(n_splits=K, shuffle=True, random_state=42)

metrics = []

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, y_train = base_model.balance_train(X.iloc[train_index], y.iloc[train_index])
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=attributes['n_estimators'], min_samples_leaf=attributes['min_samples_leaf'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.append(base_model.get_metrics(y_test, y_pred))
    print('{} done'.format(i))

with open(base_model.get_output_filepath('random_forest'), 'w') as file:
    file.write(json.dumps(metrics, indent=4))

print('FINISHED after {} seconds'.format(time.time() - start))
