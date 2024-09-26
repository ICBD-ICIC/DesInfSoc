import base_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

# param_grid = {
#     'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
#     'fit_prior': [True, False],
#     'force_alpha': [True]
# }


model = MultinomialNB(**base_model.get_best_params('naive_bayes'))
# random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=6, cv=5, random_state=42, n_jobs=-1)
# random_search.fit(X_train, y_train)
model.fit(X_train, y_train)
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

# metrics = base_model.get_metrics(y_test, y_pred)
base_model.save_confusion_matrix(y_test, y_pred, 'naive_bayes')

# with open(base_model.get_output_filepath('naive_bayes'), 'w') as file:
#      # file.write(json.dumps(metrics, indent=4))
#      # file.write(str(best_model.get_params()))
#      # file.write(str(random_search.best_params_))
#      file.write(json.dumps(confusion_matrix))

print('FINISHED after {} seconds'.format(time.time()-start))
