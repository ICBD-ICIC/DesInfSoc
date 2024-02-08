import base_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

param_grid = {
    'alpha': [0, 0.05, 0.1, 0.15, 0.2, 100],
    'fit_prior': [True, False],
    'force_alpha': [True]
}

model = MultinomialNB()
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=6, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('multinomial_naive_bayes'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(best_model.get_params()))
     file.write(str(random_search.best_params_))

print('FINISHED after {} seconds'.format(time.time()-start))
