import base_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

param_grid = {
     'hidden_layer_sizes': [(10,), (15,), (20,), (30,), (20, 20), (25, 25), (30, 30), (35, 35)],
     'batch_size': [1, 10, 50, 100, 200],
     'max_iter': [500]
}

model = MLPClassifier()
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('neural_network'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(best_model.get_params()))
     file.write(str(random_search.best_params_))

print('FINISHED after {} seconds'.format(time.time()-start))

