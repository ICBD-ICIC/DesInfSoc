import os
import ast
import json
from collections import defaultdict


# Create recursive default dicts
def fix(f):
    return lambda *args, **kwargs: f(fix(f), *args, **kwargs)


PARENT_FOLDER = 'multiclass/'
all_items = os.listdir(PARENT_FOLDER)
results_subfolders = [PARENT_FOLDER + item for item in all_items if item.startswith('results')]
file_list = []
for folder_path in results_subfolders:
    file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename), os.listdir(folder_path)))

all_best_params = fix(defaultdict)()

for file_path in file_list:
    file_name = file_path.split('/')[-1]
    dataset = file_name.split(',')[0]
    model = file_name.split(',')[1]
    feature = file_name.split(',')[2]

    file_content = open(file_path, "r").read()
    # Only save models with hyperparameters search
    if len(file_content.split('}{')) == 3:
        best_params = '{' + file_content.split('}{')[2]
        all_best_params[dataset][model][feature] = ast.literal_eval(best_params)

with open('best_params.json', 'w') as f:
    json.dump(all_best_params, f)
