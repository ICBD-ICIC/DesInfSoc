import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

FOLDER_PATHS = ['multiclass/results-all-hyperparameters-balanced(only-action)/',
                'multiclass/results-all-hyperparameters-balanced(end-to-end)/']

# Metrics that will be plotted and saved as PDF
METRICS = ['f1_micro', 'f1_macro', 'f1_weighted']

HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.']

# Translations
EXPERIMENT_TYPE_LABELS = {
    'end-to-end': 'Integral',
    'only-action': 'Solo acción',
}

COL_ENFOQUE = 'Enfoque'
COL_DISPERSION = 'Dispersión'

# Tamaños de fuente globales (sube estos valores para agrandar aún más el texto)
FONT_SIZE_TITLE = 14
FONT_SIZE_TICKS = 14
FONT_SIZE_LEGEND = 18


def comma_formatter(x, pos):
    return f"{x:.1f}".replace('.', ',')


def paper_experiments(dataframe):
    predictions = {'predominant_emotion': 'P5',
                   'predominant_sentiment': 'P4',
                   'valence_ratio': 'P6',
                   'abusive_ratio_interval': 'P1',
                   'mfd_ratio': 'P3',
                   'polarization_ratio_interval': 'P2'}
    df = dataframe
    df['prediction'] = df['prediction'].replace(predictions)
    df['spread'] = df['spread'].str[-2:]

    # Map experiment_type values to their Spanish labels instead of the
    # generic "replace dashes + capitalize" transformation
    df['experiment_type'] = df['experiment_type'].map(EXPERIMENT_TYPE_LABELS).fillna(
        df['experiment_type'].str.replace('-', ' ').str.capitalize()
    )

    df = df.sort_values(by=['spread', 'prediction', 'experiment_type'], ascending=[False, True, True])

    # Rename columns for display purposes (facet titles / axis labels)
    df = df.rename(columns={'experiment_type': COL_ENFOQUE, 'spread': COL_DISPERSION})
    return df


file_list = []
for folder_path in FOLDER_PATHS:
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
        model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                      'prediction': prediction, 'spread': spread, 'experiment_type': experiment_type}
        plot_data.append(model_data)

full_dataframe = pd.DataFrame(plot_data)
full_dataframe = paper_experiments(full_dataframe)

sns.set_style("whitegrid")
sns.set_palette("pastel")
# Escala el tamaño de fuente base de todos los elementos (labels, ticks, etc.)
sns.set_context("talk", rc={"font.size": FONT_SIZE_TICKS,
                             "axes.titlesize": FONT_SIZE_TITLE,
                             "axes.labelsize": FONT_SIZE_TICKS,
                             "xtick.labelsize": FONT_SIZE_TICKS,
                             "ytick.labelsize": FONT_SIZE_TICKS,
                             "legend.fontsize": FONT_SIZE_LEGEND})

os.makedirs('plots', exist_ok=True)


def save_legend_separately(legend_patches, output_path):
    """Genera un PDF aparte solo con la leyenda, en horizontal (una fila)."""
    fig_legend = plt.figure(figsize=(max(6, 1.6 * len(legend_patches)), 1.0))
    fig_legend.legend(handles=legend_patches, ncol=len(legend_patches), fontsize=FONT_SIZE_LEGEND,
                       loc='center', labelspacing=0.4,
                       handlelength=1.5, handletextpad=0.5, borderpad=0.4,
                       frameon=False)
    fig_legend.savefig(output_path, bbox_inches='tight')
    plt.close(fig_legend)


def plot_metric(dataframe, metric_name, save_legend=False):
    df = dataframe.loc[dataframe['metric_name'] == metric_name]

    p = sns.catplot(data=df, x="prediction", y="metric_value", hue="model", col=COL_DISPERSION,
                     kind="bar", edgecolor='black', row=COL_ENFOQUE, legend=False, height=2.5, aspect=1.5)
    p.set(xlabel=None, ylabel=None)
    p.set_titles(row_template="Enfoque: {row_name}", col_template="Dispersión: {col_name}", size=FONT_SIZE_TITLE)

    for axes in p.axes.flat:
        axes.tick_params(axis='both', labelsize=FONT_SIZE_TICKS)

    colors = []
    models_names = list(map(lambda name: name.replace('_', ' ').capitalize(), df['model'].unique()))

    for axes in p.axes.flat:
        for bar in axes.patches:
            bar_color = bar.get_facecolor()
            colors.append(bar_color) if bar_color not in colors else colors
            bar.set_hatch(HATCHES[colors.index(bar_color)])

    legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=models_names[i])
                      for i in range(0, len(colors))]

    # La leyenda ya NO se añade a la figura principal
    p.fig.set_constrained_layout(True)  # gestiona automáticamente los títulos sin solapes

    plt.yticks(np.arange(0, 1.1, 0.1))

    for axes in p.axes.flat:
        axes.yaxis.set_major_formatter(FuncFormatter(comma_formatter))

    output_path = f'plots/{metric_name}.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # La leyenda es la misma para todas las métricas (mismos modelos), así que
    # solo hace falta guardarla una vez como plots/legend.pdf
    if save_legend:
        save_legend_separately(legend_patches, 'plots/legend.pdf')


for i, metric in enumerate(METRICS):
    plot_metric(full_dataframe, metric, save_legend=(i == 0))