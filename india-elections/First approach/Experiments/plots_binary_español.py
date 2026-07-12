import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

DATASET_FOLDER_PATHS = {
    'balanced': ['binary/results-all-hyperparameters-balanced(only-action)/',
                 'binary/results-all-hyperparameters-balanced(end-to-end)/'],
    'imbalanced': ['binary/results-all-hyperparameters-imbalanced(only-action)/',
                   'binary/results-all-hyperparameters-imbalanced(end-to-end)/'],
}

METRIC = 'f1'  # <-- cambia esto si la clave en tus JSON se llama distinto (p. ej. 'f1_score')

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']

EXPERIMENT_TYPE_LABELS = {
    'end-to-end': 'Integral',
    'only-action': 'Solo acción',
}

COL_ENFOQUE = 'Enfoque'
COL_DISPERSION = 'Dispersión'


def comma_formatter(x, pos):
    return f"{x:.1f}".replace('.', ',')


def paper_experiments(dataframe):
    predictions = {'predominant_emotion': 'P5',
                   'predominant_sentiment': 'P4',
                   'valence_ratio': 'P6',
                   'abusive_ratio_interval': 'P1',
                   'mfd_ratio': 'P3',
                   'polarization_ratio_interval': 'P2'}
    df = dataframe.copy()
    df['prediction'] = df['prediction'].replace(predictions)
    df['spread'] = df['spread'].str[-2:]
    df = df[df['spread'].isin(['20', '60'])]

    df['experiment_type'] = df['experiment_type'].map(EXPERIMENT_TYPE_LABELS).fillna(
        df['experiment_type'].str.replace('-', ' ').str.capitalize()
    )

    df = df.sort_values(by=['spread', 'prediction', 'experiment_type'], ascending=[False, True, True])
    df = df.rename(columns={'experiment_type': COL_ENFOQUE, 'spread': COL_DISPERSION})
    return df


def load_dataframe(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            file_list.append(file_path)

    plot_data = []
    for file in file_list:
        metrics = open(file, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)
        spread = file.split(',')[0].split('_')[1].replace('SPREAD', '')
        model = file.split(',')[1]
        prediction = file.split(',')[2]
        experiment_type = file.split('(')[1].split(')')[0]
        for name, value in metrics.items():
            plot_data.append({'metric_name': name, 'metric_value': value, 'model': model,
                               'prediction': prediction, 'spread': spread, 'experiment_type': experiment_type})

    dataframe = pd.DataFrame(plot_data)
    dataframe = paper_experiments(dataframe)
    return dataframe


def plot_metric(dataframe, metric_name, output_path):
    df = dataframe.loc[dataframe['metric_name'] == metric_name]

    p = sns.catplot(data=df, x="prediction", y="metric_value", hue="model", col=COL_DISPERSION,
                     kind="bar", edgecolor='black', row=COL_ENFOQUE, legend=False, height=2.5, aspect=1.5)
    p.fig.set_size_inches(9, 4.5)
    p.set(xlabel=None, ylabel=None)
    p.set_titles(row_template="Enfoque: {row_name}", col_template="Dispersión: {col_name}", size=10)

    colors = []
    models_names = list(map(lambda name: name.replace('_', ' ').capitalize(), df['model'].unique()))

    for axes in p.axes.flat:
        for bar in axes.patches:
            bar_color = bar.get_facecolor()
            colors.append(bar_color) if bar_color not in colors else colors
            bar.set_hatch(HATCHES[colors.index(bar_color)])

    legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=models_names[i])
                      for i in range(0, len(colors))]

    p.fig.set_constrained_layout(True)
    p.fig.legend(handles=legend_patches, ncol=1, fontsize=10,
                 loc='outside center right', labelspacing=0.4,
                 handlelength=1.5, handletextpad=0.5, borderpad=0.4)

    plt.yticks(np.arange(0, 1.1, 0.1))

    for axes in p.axes.flat:
        axes.yaxis.set_major_formatter(FuncFormatter(comma_formatter))

    plt.savefig(output_path)
    plt.close()


sns.set_style("whitegrid")
sns.set_palette("pastel")

os.makedirs('plots', exist_ok=True)

for dataset_type, folder_paths in DATASET_FOLDER_PATHS.items():
    dataframe = load_dataframe(folder_paths)
    output_path = f'plots/f1_{dataset_type}.pdf'
    plot_metric(dataframe, METRIC, output_path)
    print(f'Guardado: {output_path}')