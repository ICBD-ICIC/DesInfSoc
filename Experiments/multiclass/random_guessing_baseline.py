import base_model
import json


def fbeta(beta, precision, recall):
    return (1 + (beta ** 2)) * ((precision * recall) / ((beta ** 2) * precision + recall))


X, y = base_model.get_dataset()
class_amounts = y.value_counts()

total = len(y)
classes_names = y.value_counts().index.values
class_totals = y.value_counts().values

probabilities = list(map(lambda class_total: class_total / total, class_totals))

TPs = []
TNs = []
FPs = []
FNs = []
precisions = []
recalls = []
accuracies = []
f1s = []
fbetas = {}
for beta_option in base_model.BETA_OPTIONS:
    fbetas[beta_option] = []

for i, class_name in enumerate(classes_names):
    TP = probabilities[i] * class_totals[i]
    TN = (1 - probabilities[i]) * (total - class_totals[i])
    FP = probabilities[i] * (total - class_totals[i])
    FN = (1 - probabilities[i]) * class_totals[i]

    TPs.append(TP)
    TNs.append(TN)
    FPs.append(FP)
    FNs.append(FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / total
    f1 = fbeta(1, precision, recall)

    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    f1s.append(f1)

    for beta_option in base_model.BETA_OPTIONS:
        fbetas[beta_option].append(fbeta(beta_option, precision, recall))

print(TPs)
print(TNs)
print(FPs)
print(FNs)
print(recalls)
print(precisions)

#########################
######### MACRO #########
#########################

metrics = {'precision_macro': sum(precisions) / len(precisions),
           'recall_macro': sum(recalls) / len(recalls),
           'f1_macro': sum(f1s) / len(f1s)}
for beta_option in base_model.BETA_OPTIONS:
    metrics['fbeta_{}_macro'.format(beta_option)] = sum(fbetas[beta_option]) / len(fbetas[beta_option])

########################
####### WEIGHTED #######
########################

precision_weighted = 0
recall_weighted = 0
f1_weighted = 0
fbetas_weighted = {}
for beta_option in base_model.BETA_OPTIONS:
    fbetas_weighted[beta_option] = 0

for i, class_name in enumerate(classes_names):
    precision_weighted += (precisions[i] * (class_totals[i] / total))
    recall_weighted += (recalls[i] * (class_totals[i] / total))
    f1_weighted += (f1s[i] * (class_totals[i] / total))
    for beta_option in base_model.BETA_OPTIONS:
        fbetas_weighted[beta_option] += (fbetas[beta_option][i] * (class_totals[i] / total))

metrics['precision_weighted'] = precision_weighted
metrics['recall_weighted'] = recall_weighted
metrics['f1_weighted'] = f1_weighted
for beta_option in base_model.BETA_OPTIONS:
    metrics['fbeta_{}_weighted'.format(beta_option)] = fbetas_weighted[beta_option]

#########################
######### MICRO #########
#########################

precision_micro = sum(TPs) / (sum(TPs) + sum(FPs))
recall_micro = sum(TPs) / (sum(TPs) + sum(FNs))
f1_micro = fbeta(1, precision_micro, recall_micro)

metrics['precision_micro'] = precision_micro
metrics['recall_micro'] = recall_micro
metrics['f1_micro'] = f1_micro

for beta_option in base_model.BETA_OPTIONS:
    metrics['fbeta_{}_micro'] = fbeta(beta_option, precision_micro, recall_micro)

with open(base_model.get_output_filepath('random_guessing'), 'w') as file:
    file.write(json.dumps(metrics, indent=4))
