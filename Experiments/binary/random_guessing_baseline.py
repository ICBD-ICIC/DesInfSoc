import base_model
import json


def fbeta(beta, precision, recall):
    return (1 + (beta ** 2)) * ((precision * recall) / ((beta ** 2) * precision + recall))


X, y = base_model.get_dataset()
class_amounts = y.value_counts()

total = class_amounts[0] + class_amounts[1]
total_negative = class_amounts[0]
total_positive = class_amounts[1]

probability_positive = total_positive / total
probability_negative = total_negative / total

TP = probability_positive * total_positive
TN = probability_negative * total_negative
FP = probability_positive * total_negative
FN = probability_negative * total_positive

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / total
f1 = fbeta(1, precision, recall)
precision_recall_auc = recall  # y * (1-0)

metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': 0.5,
           'precision_recall_auc': precision_recall_auc}

for beta_option in base_model.BETA_OPTIONS:
    metrics['fbeta_{}'.format(beta_option)] = fbeta(beta_option, precision, recall)

with open(base_model.get_output_filepath('random_guessing'), 'w') as file:
    file.write(json.dumps(metrics, indent=4))
