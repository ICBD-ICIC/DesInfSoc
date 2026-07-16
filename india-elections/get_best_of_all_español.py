import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

pd.set_option("max_colwidth", 200)
pd.set_option("display.max_columns", None)

# Tamaños de fuente globales (sube estos valores para agrandar aún más el texto)
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS = 16
FONT_SIZE_TICKS = 14
FONT_SIZE_BAR_LABELS = 14

APPROACH_LABELS = {
    'end-to-end': 'I',   # Integral
    'only-action': 'SA', # Solo acción
}

MODEL_LABELS = {
    'Random guessing': 'RG',
    'Random forest': 'RF',
    'Neural network': 'NN',
    'Naive bayes': 'NB',
    'Multinomial naive bayes': 'MNB',
    'Logistic regression': 'LR',
    'Decision tree': 'DT',
    'Complement naive bayes': 'CNB',
    'Nn 2 hidden layers': 'NN 2',
    'Nn 2 hidden layers 35': 'NN 2-35',
}

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

METRIC_LABELS = {
    'f1': 'F1',
    'f1_micro': 'F1 micro',
    'f1_macro': 'F1 macro',
    'f1_weighted': 'F1 ponderado',
}

def coma_decimal(x, pos):
    return f'{x:.2f}'.replace('.', ',')

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
            words_present = 'Falso'
        else:
            words_present = 'Verdadero'
        if 'imbalanced' in file_path:
            class_balance = 'Falso'
        else:
            class_balance = 'Verdadero'
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
    metric_label = METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').capitalize())
    approach_label = APPROACH_LABELS.get(best['approach'], best['approach'].replace('-', ' ').capitalize())
    words_present_label = best['words_present'][0]   # Falso -> F, Verdadero -> V
    class_balance_label = best['class_balance'][0]   # Falso -> F, Verdadero -> V
    return (f"Mejor puntuación de {metric_label} para {prediction}",
            f"D: {best['spread']} | E: {approach_label} | "
            f"P: {words_present_label} | B: {class_balance_label}")


def get_plot(df, best_idx, metric_name, metrics_names):
    columns_to_drop = metrics_names + ['file_path', 'model']
    filtering_criteria = df.iloc[best_idx].drop(columns_to_drop)

    filtering_condition = pd.Series([True] * len(df))
    for col, val in filtering_criteria.items():
        filtering_condition &= (df[col] == val)

    filtered_df = df[filtering_condition].copy()
    filtered_df['model_label'] = filtered_df['model'].map(MODEL_LABELS).fillna(filtered_df['model'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(filtered_df['model_label'], filtered_df[metric_name])
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad=5, labelsize=FONT_SIZE_TICKS)
    ax.yaxis.set_tick_params(pad=10, labelsize=FONT_SIZE_TICKS)
    ax.grid(color='black',
            linestyle='-', linewidth=1,
            alpha=0.2)
    metric_label = METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').capitalize())
    ax.set_xlabel(metric_label, fontsize=FONT_SIZE_AXIS)
    title, subtitle = get_plot_title(df.iloc[best_idx], metric_name)
    ax.set_title(subtitle, loc='center', fontsize=FONT_SIZE_TITLE)
    max_value = round(filtered_df[metric_name].max(), 2)
    for bar in bars:
        width = round(bar.get_width(), 2)
        if width == max_value:
            bar.set_color('red')
        ax.text(width + 0.01 * width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}'.replace('.', ','),
                va='center', ha='left', fontsize=FONT_SIZE_BAR_LABELS)
    ax.set_xlim(0, 1.08)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=FONT_SIZE_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(coma_decimal))
    plt.tight_layout()
    plt.savefig(f"best_results_plots/{df.iloc[best_idx]['prediction']}_{metric_name}.pdf",
                bbox_inches='tight')
    plt.close(fig)

os.makedirs('best_results_plots', exist_ok=True)

# BINARIO #
print('BINARIO')
metrics_names = ['f1']
df = get_metrics_values(get_all_results_files(BINARY_FOLDERS), metrics_names)
for prediction in df['prediction'].unique():
    best_result_idx = df.loc[df['prediction'] == prediction]['f1'].idxmax()
    get_plot(df, best_result_idx, 'f1', metrics_names)

# MULTICLASE #
print('MULTICLASE')
metrics_names = ['f1_micro', 'f1_macro', 'f1_weighted']
df = get_metrics_values(get_all_results_files(MULTICLASS_FOLDERS), metrics_names)
for prediction in df['prediction'].unique():
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_micro'].idxmax()
    get_plot(df, best_result_idx, 'f1_micro', metrics_names)
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_macro'].idxmax()
    get_plot(df, best_result_idx, 'f1_macro', metrics_names)
    best_result_idx = df.loc[df['prediction'] == prediction]['f1_weighted'].idxmax()
    get_plot(df, best_result_idx, 'f1_weighted', metrics_names)

# # Mejor sentimiento y emoción descartando Random Guessing
# metrics_names = ['f1_micro', 'f1_weighted', 'f1_macro']
# predictions = ['predominant_sentiment', 'predominant_emotion']
# df = get_metrics_values(get_all_results_files(MULTICLASS_FOLDERS), metrics_names)
# filtered_df = df[df['model'] != 'Random guessing']
# for prediction in predictions:
#     best_result_idx = filtered_df.loc[filtered_df['prediction'] == prediction]['f1_micro'].idxmax()
#     get_plot(df, best_result_idx, 'f1_micro', metrics_names)
#     best_result_idx = filtered_df.loc[filtered_df['prediction'] == prediction]['f1_weighted'].idxmax()
#     get_plot(df, best_result_idx, 'f1_weighted', metrics_names)