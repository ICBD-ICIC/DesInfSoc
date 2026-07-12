from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

FOLDER_PATHS = ['binary/feature-selection/results-t_test-30_runs(spread20,balanced)/only-action/',
                'binary/feature-selection/results-t_test-30_runs(spread20,balanced)/end-to-end/']
PLOTS_FOLDER = 'plots'

# Métrica -> (nombre en español para título, nombre de archivo)
METRICS_INFO = {
    'f1': {'title': 'F1 - promedio de 30 ejecuciones', 'filename': 'f1_score_promedio_30_ejecuciones.pdf'},
    'precision': {'title': 'Precisión - promedio de 30 ejecuciones', 'filename': 'precision_promedio_30_ejecuciones.pdf'},
    'recall': {'title': 'Exhaustividad - promedio de 30 ejecuciones', 'filename': 'exhaustividad_promedio_30_ejecuciones.pdf'},
}

HATCHES = ['\\\\', '-', '//', '..', '', '++', 'oo', '||', 'XX', 'OO', '\\', '--', '/', '.', '+', 'o', '|', 'X', 'o']

def coma_decimal(x, pos):
    return f'{x:.2f}'.replace('.', ',')

def paper_experiment_one(dataframe):
    features = {'emotion': 'Solo emoción',
                'emotion+linguistic-ratio': 'Emoción + Proporción Lingüística',
                'personality': 'Solo personalidad',
                'linguistic-ratio+personality': 'Personalidad + Proporción Lingüística',
                'sentiment': 'Solo sentimiento',
                'linguistic-ratio+sentiment': 'Sentimiento + Proporción Lingüística',
                'emotion+personality': 'Emoción + Personalidad',
                'emotion+linguistic-ratio+personality': 'Emoción + Personalidad + Proporción Lingüística',
                'emotion+sentiment': 'Emoción + Sentimiento',
                'emotion+linguistic-ratio+sentiment': 'Emoción + Sentimiento + Proporción Lingüística',
                'personality+sentiment': 'Personalidad + Sentimiento',
                'linguistic-ratio+personality+sentiment': 'Personalidad + Sentimiento + Proporción Lingüística',
                'emotion+personality+sentiment': 'Emoción + Sentimiento + Personalidad',
                'emotion+linguistic-ratio+personality+sentiment': 'Emoción + Sentimiento + Personalidad + Proporción Lingüística',
                'emotion+linguistic-ratio+personality+sentiment+linguistic-amount': 'Todas'
                }
    df = dataframe.copy()
    df = df.loc[dataframe['feature'].isin(features.keys())]
    df['feature'] = pd.Categorical(df['feature'], categories=features.keys(), ordered=True)
    df['feature'] = df['feature'].cat.rename_categories(features)

    predictions = {'abusive_ratio_interval': 'P1',
                   'polarization_ratio_interval': 'P2',
                   'mfd_ratio': 'P3',
                   'predominant_sentiment': 'P4',
                   'predominant_emotion': 'P5',
                   'valence_ratio': 'P6'}
    df['prediction'] = df['prediction'].replace(predictions)

    df = df.sort_values(by=['feature', 'prediction'])

    df['approach'] = df['approach'].str.replace('-', ' ').str.capitalize()
    approach_translations = {'End to end': 'Enfoque: Integral',
                              'Only action': 'Enfoque: Solo acción'}
    df['approach'] = df['approach'].replace(approach_translations)

    return df


def build_plot_dataframe():
    plot_data = []

    experiment_subfolders = []
    for folder_path in FOLDER_PATHS:
        all_items = os.listdir(folder_path)
        experiment_subfolders.extend([folder_path + item for item in all_items if item.startswith('experiments')])

    file_list = []
    for folder_path in experiment_subfolders:
        file_list += list(map(lambda filename: '{}/{}'.format(folder_path, filename), os.listdir(folder_path)))

    for file in file_list:
        k_metrics = open(file, "r").read()
        k_metrics = json.loads(k_metrics)
        feature = file.split('/')[4].replace('experiments-', '')
        prediction = file.split('/')[5].split(',')[2]
        f1_values = []
        recall_values = []
        precision_values = []
        approach = file.split('/')[3]
        for instance in k_metrics:
            f1_values.append(instance['f1'])
            recall_values.append(instance['recall'])
            precision_values.append(instance['precision'])
        plot_data.append({'metric_name': 'f1', 'metric_value': mean(f1_values), 'prediction': prediction,
                          'feature': feature, 'approach': approach})
        plot_data.append({'metric_name': 'recall', 'metric_value': mean(recall_values), 'prediction': prediction,
                          'feature': feature, 'approach': approach})
        plot_data.append({'metric_name': 'precision', 'metric_value': mean(precision_values), 'prediction': prediction,
                          'feature': feature, 'approach': approach})

    dataframe = pd.DataFrame(plot_data)
    dataframe = paper_experiment_one(dataframe)
    return dataframe


def make_plot(dataframe, metric, title, filename):
    df_metric = dataframe.loc[dataframe['metric_name'] == metric]

    for prediction in df_metric['prediction'].unique():
        best_result_idx = df_metric.loc[(df_metric['prediction'] == prediction) & (df_metric['approach'] == 'Enfoque: Integral')][
            'metric_value'].idxmax()
        print(metric, prediction, 'enfoque-integral', df_metric.loc[best_result_idx]['feature'])
        best_result_idx = df_metric.loc[(df_metric['prediction'] == prediction) & (df_metric['approach'] == 'Enfoque: Solo acción')][
            'metric_value'].idxmax()
        print(metric, prediction, 'enfoque-solo-accion', df_metric.loc[best_result_idx]['feature'])

    sns.set_style("whitegrid")

    p = sns.catplot(data=df_metric, x="prediction", y="metric_value", hue="feature", kind="bar", col='approach',
                    edgecolor='black', legend=False, height=6, aspect=1.1)
    p.set(xlabel=None, ylabel=None)
    p.set_axis_labels('', title)
    p.set_titles("{col_name}")

    colors = []
    features = df_metric['feature'].unique()

    for axes in p.axes.flat:
        axes.set_title(axes.get_title())
        axes.yaxis.set_major_formatter(FuncFormatter(coma_decimal))
        for bar in axes.patches:
            bar_color = bar.get_facecolor()
            colors.append(bar_color) if bar_color not in colors else colors
            bar.set_hatch(HATCHES[colors.index(bar_color)])

    legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=features[i])
                      for i in range(0, len(colors))]

    p.fig.legend(handles=legend_patches, loc='lower center', ncols=3, fontsize=10,
                 bbox_to_anchor=(0.5, -0.08), frameon=True)

    plt.suptitle('Dispersión: 20 | Palabras presentes: Falso | Balanceado: Verdadero', size=16)
    plt.subplots_adjust(top=0.88, bottom=0.18)

    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    output_path = os.path.join(PLOTS_FOLDER, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()


dataframe = build_plot_dataframe()

for metric, info in METRICS_INFO.items():
    make_plot(dataframe, metric, info['title'], info['filename'])