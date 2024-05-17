import base_model
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

param_grid = {
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

model = OneClassSVM()
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=6, cv=5, random_state=42, n_jobs=-1,
                                   scoring='accuracy')
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

print(set(y_test) - set(y_pred))

metrics = base_model.get_metrics(y_test, y_pred)

with open(base_model.get_output_filepath('support_vector_machine'), 'w') as file:
     file.write(json.dumps(metrics, indent=4))
     file.write(str(best_model.get_params()))
     file.write(str(random_search.best_params_))

print('FINISHED after {} seconds'.format(time.time()-start))

