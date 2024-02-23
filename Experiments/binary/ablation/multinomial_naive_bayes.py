import base_model
from sklearn.naive_bayes import MultinomialNB
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

BEST_ATTRIBUTES_SPREAD20 = {
    '28': {'fit_prior': False, 'alpha': 0.2},
    '30': {'fit_prior': False, 'alpha': 0.2},
    '31': {'fit_prior': False, 'alpha': 0.2},
    '32': {'fit_prior': False, 'alpha': 0.2},
    '34': {'fit_prior': True, 'alpha': 0},
    '36': {'fit_prior': False, 'alpha': 0.2}
}

BEST_ATTRIBUTES_SPREAD60 = {
    '28': {'fit_prior': False, 'alpha': 0.2},
    '30': {'fit_prior': False, 'alpha': 0.2},
    '31': {'fit_prior': False, 'alpha': 0.2},
    '32': {'fit_prior': False, 'alpha': 0.2},
    '34': {'fit_prior': True, 'alpha': 0},
    '36': {'fit_prior': False, 'alpha': 0.2}
}

BEST_ATTRIBUTES = {
    'context_SPREAD60_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD60,
    'context_SPREAD20_K3_H4_P12-BINARY': BEST_ATTRIBUTES_SPREAD20
}

attributes = BEST_ATTRIBUTES[base_model.dataset_name][base_model.prediction]


model = MultinomialNB(force_alpha=True, alpha=attributes['alpha'], fit_prior=attributes['fit_prior'])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('multinomial_naive_bayes'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(model.get_params()))

print('FINISHED after {} seconds'.format(time.time()-start))
