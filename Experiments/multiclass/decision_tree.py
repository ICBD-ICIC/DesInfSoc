import base_model
from sklearn.tree import DecisionTreeClassifier
import time
import json

start = time.time()

X_train, X_test, y_train, y_test = base_model.get_train_test_split()

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(set(y_test) - set(y_pred))

# metrics = base_model.get_metrics(y_test, y_pred)
#
# with open(base_model.get_output_filepath('decision_tree'), 'w') as file:
#      file.write(json.dumps(metrics, indent=4))
#      file.write(str(model.get_params()))

base_model.save_confusion_matrix(y_test, y_pred, 'decision_tree')

print('FINISHED after {} seconds'.format(time.time()-start))
