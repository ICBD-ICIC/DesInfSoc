import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

EXPERIMENT_TYPES = ['distance', 'pattern_matching']
METRICS = ['precision', 'recall', 'f1']

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']

EXPERIMENT_TYPE_LABELS = {
    'distance': 'Distancia de palabras',
    'pattern_matching': 'Coincidencia de patrones',
}

METRIC_LABELS = {
    'precision': 'Precisión',
    'recall': 'Exhaustividad',
    'f1': 'F1',
}

LABEL_PREDICTION_MAP = {
    "abusive_ratio_interval_gt": "P1",
    "polarization_ratio_interval_gt": "P2",
    "mfd_ratio_gt": "P3",
    "predominant_sentiment_gt": "P4",
    "predominant_emotion_gt": "P5",
    "valence_ratio_gt": "P6",
}


def comma_formatter(x, pos):
    return f"{x:.1f}".replace('.', ',')


def load_dataframe(experiment_type):
    """Carga y normaliza los datos de un tipo de experimento (equivalente a
    load_dataframe en el script de referencia, pero para la estructura de
    carpetas experiments-{experiment_type})."""
    folder_paths = [f'../experiments/experiments-{experiment_type}/']

    file_list = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            file_list.append(os.path.join(folder_path, file))

    plot_data = []
    for filename in file_list:
        metrics = open(filename, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)

        model = filename.split(',')[1]
        prediction = filename.split(',')[2]

        for name, value in metrics.items():
            plot_data.append({
                'metric_name': name,
                'metric_value': value,
                'model': model.replace('_', ' ').title(),
                'prediction': LABEL_PREDICTION_MAP[prediction],
            })

    dataframe = pd.DataFrame(plot_data)
    dataframe = dataframe.sort_values(by=['prediction', 'model'])
    return dataframe


def save_legend_separately(legend_patches, output_path):
    """Genera un PDF aparte solo con la leyenda, en una fila horizontal."""
    fig_legend = plt.figure(figsize=(max(6, 1.8 * len(legend_patches)), 1.2))
    fig_legend.legend(handles=legend_patches, ncol=len(legend_patches), fontsize=20,
                       loc='center', labelspacing=0.4,
                       handlelength=1.5, handletextpad=0.5, borderpad=0.4,
                       frameon=False)
    fig_legend.savefig(output_path, bbox_inches='tight')
    plt.close(fig_legend)


def plot_metric(metric_name, output_path, save_legend=False):
    """Genera la figura para una métrica concreta, con un subplot por cada
    EXPERIMENT_TYPE (lado a lado, sin espacio entre ellos), usando hatches
    para distinguir modelos. La leyenda se guarda por separado."""
    fig = plt.figure(figsize=(12, 4))
    spec = gridspec.GridSpec(
        ncols=len(EXPERIMENT_TYPES),
        nrows=1,
        width_ratios=[1] * len(EXPERIMENT_TYPES),
        wspace=0
    )
    axes = [fig.add_subplot(spec[i]) for i in range(len(EXPERIMENT_TYPES))]

    colors = []
    legend_labels = []

    for idx, experiment_type in enumerate(EXPERIMENT_TYPES):
        dataframe = load_dataframe(experiment_type)
        df = dataframe.loc[dataframe['metric_name'] == metric_name]

        sns.barplot(
            data=df,
            x="prediction",
            y="metric_value",
            hue="model",
            ax=axes[idx],
            edgecolor='black'
        )

        handles, labels = axes[idx].get_legend_handles_labels()
        if not legend_labels:
            legend_labels = labels

        for bar in axes[idx].patches:
            bar_color = bar.get_facecolor()
            if bar_color not in colors:
                colors.append(bar_color)
            bar.set_hatch(HATCHES[colors.index(bar_color)])

        axes[idx].set_title(
            EXPERIMENT_TYPE_LABELS.get(experiment_type, experiment_type.replace('_', ' ').title()),
            fontsize=20
        )
        axes[idx].tick_params(axis='x', labelsize=11)
        axes[idx].set_xlabel("")

        if idx == 0:
            axes[idx].set_ylabel(METRIC_LABELS.get(metric_name, metric_name.capitalize()), fontsize=20)
            axes[idx].set_yticks(np.arange(0, 1.1, 0.1))
            axes[idx].yaxis.set_major_formatter(FuncFormatter(comma_formatter))
        else:
            axes[idx].set_ylabel("")
            axes[idx].set_yticks(np.arange(0, 1.1, 0.1))  # mantiene las líneas de la grilla
            axes[idx].set_yticklabels([])  # oculta los números
            axes[idx].tick_params(axis='y', left=False)  # oculta las marcas

        axes[idx].tick_params(labelsize=20)
        axes[idx].get_legend().remove()

    legend_patches = [
        Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=legend_labels[i])
        for i in range(len(colors))
    ]

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    if save_legend:
        save_legend_separately(legend_patches, 'plots/legend.pdf')


sns.set_style("whitegrid")
sns.set_palette("colorblind")

os.makedirs('plots', exist_ok=True)

for i, METRIC in enumerate(METRICS):
    output_path = f'plots/{METRIC}.pdf'
    plot_metric(METRIC, output_path, save_legend=(i == 0))
    print(f'Guardado: {output_path}')