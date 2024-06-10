import json
import os

import pandas as pd
from matplotlib import pyplot as plt

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

BINARY_FOLDERS = [
    'Experiments/binary/experiments(spread20,balanced,only-action)',
    'Experiments/binary/experiments(spread100,imbalanced,only-action)',
    'First approach/Experiments/binary/results-all-hyperparameters-balanced(end-to-end)',
    'First approach/Experiments/binary/results-all-hyperparameters-balanced(only-action)',
    'First approach/Experiments/binary/results-all-hyperparameters-imbalanced(end-to-end)',
    'First approach/Experiments/binary/results-all-hyperparameters-imbalanced(only-action)',
]
MULTICLASS_FOLDERS = [
    'First approach/Experiments/multiclass/results-all-hyperparameters-balanced(end-to-end)',
    'First approach/Experiments/multiclass/results-all-hyperparameters-balanced(only-action)'
]


def get_all_results_files(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            file_list.append(file_path)
    return file_list


def get_metrics_values(file_list, metrics_list):
    metrics_values = []
    print('Total', len(file_list))

    for file_path in file_list:
        metrics = open(file_path, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)
        filename = file_path.replace('\\', '/').split('/')[-1]
        model = filename.split(',')[1]
        prediction = filename.split(',')[2]
        spread = filename.split(',')[0].split('_')[1].split('-')[-1].replace('SPREAD', '')
        if len(filename.split(',')[0].split('_')[1].split('-')) > 1:
            approach = 'only-action'
        else:
            approach = 'end-to-end'
        if 'First approach' in file_path:
            words_present = False
        else:
            words_present = True
        if 'imbalanced' in file_path:
            class_balance = False
        else:
            class_balance = True
        metrics = {key: metrics[key] for key in metrics_list}
        for name, value in metrics.items():
            model_data = {f"{name}": value, 'file_path': file_path, 'prediction': prediction,
                          'model': model.replace('_', ' ').capitalize(),
                          'spread': spread, 'approach': approach, 'words_present': words_present,
                          'class_balance': class_balance}
            metrics_values.append(model_data)

    return pd.DataFrame(metrics_values)


def get_plot_title(best, metric_name):
    prediction = best['prediction'].replace('_', ' ').capitalize().replace('Mfd', 'MFD')
    return (f"Best {metric_name.replace('_', ' ').capitalize()} score for {prediction}",
            f"Spread: {best['spread']} - Approach: {best['approach'].replace('-', ' ').capitalize()} - "
            f"Words present: {best['words_present']} - Balanced: {best['class_balance']}")


def get_plot(df, best_idx, metric_name, metrics_names):
    columns_to_drop = metrics_names + ['file_path', 'model']
    print(df.loc[best_idx])
    print(df.iloc[best_idx])
    filtering_criteria = df.iloc[best_idx].drop(columns_to_drop)

    filtering_condition = pd.Series([True] * len(df))
    for col, val in filtering_criteria.items():
        filtering_condition &= (df[col] == val)

    filtered_df = df[filtering_condition]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(filtered_df['model'], filtered_df[metric_name])
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.grid(color='black',
            linestyle='-', linewidth=1,
            alpha=0.2)
    ax.set_xlabel(f"{metric_name.replace('_', ' ').capitalize()}")
    title, subtitle = get_plot_title(df.iloc[best_idx], metric_name)
    fig.suptitle(title)
    ax.set_title(subtitle, loc='center')
    plt.subplots_adjust(left=0.25, top=0.85)
    max_value = round(filtered_df[metric_name].max(), 2)
    for bar in bars:
        width = round(bar.get_width(), 2)
        if width == max_value:
            bar.set_color('red')
        ax.text(width + 0.01 * width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left')
    plt.show()


# # BINARY #
# print('BINARY')
# metrics_names = ['f1']
# df = get_metrics_values(get_all_results_files(BINARY_FOLDERS), metrics_names)
# for prediction in df['prediction'].unique():
#     best_result_idx = df.loc[df['prediction'] == prediction]['f1'].idxmax()
#     get_plot(df, best_result_idx, 'f1', metrics_names)
#
# # MULTICLASS #
# print('MULTICLASS')
# metrics_names = ['f1_micro', 'f1_macro', 'f1_weighted']
# df = get_metrics_values(get_all_results_files(MULTICLASS_FOLDERS), metrics_names)
# for prediction in df['prediction'].unique():
#     best_result_idx = df.loc[df['prediction'] == prediction]['f1_micro'].idxmax()
#     get_plot(df, best_result_idx, 'f1_micro', metrics_names)
#     best_result_idx = df.loc[df['prediction'] == prediction]['f1_macro'].idxmax()
#     get_plot(df, best_result_idx, 'f1_macro', metrics_names)
#     best_result_idx = df.loc[df['prediction'] == prediction]['f1_weighted'].idxmax()
#     get_plot(df, best_result_idx, 'f1_weighted', metrics_names)

# Best sentiment and emotion discarding Random Guessing
metrics_names = ['f1_micro', 'f1_weighted', 'f1_macro']
predictions = ['predominant_sentiment', 'predominant_emotion']
df = get_metrics_values(get_all_results_files(MULTICLASS_FOLDERS), metrics_names)
filtered_df = df[df['model'] != 'Random guessing']
for prediction in predictions:
    best_result_idx = filtered_df.loc[filtered_df['prediction'] == prediction]['f1_micro'].idxmax()
    get_plot(df, best_result_idx, 'f1_micro', metrics_names)
    best_result_idx = filtered_df.loc[filtered_df['prediction'] == prediction]['f1_weighted'].idxmax()
    get_plot(df, best_result_idx, 'f1_weighted', metrics_names)
