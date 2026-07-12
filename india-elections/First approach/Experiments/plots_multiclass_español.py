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


def plot_metric(dataframe, metric_name):
    df = dataframe.loc[dataframe['metric_name'] == metric_name]

    p = sns.catplot(data=df, x="prediction", y="metric_value", hue="model", col=COL_DISPERSION,
                     kind="bar", edgecolor='black', row=COL_ENFOQUE, legend=False, height=2.5, aspect=1.5)
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

    p.fig.set_constrained_layout(True)  # gestiona automáticamente títulos + leyenda sin solapes
    leg = p.fig.legend(handles=legend_patches, ncol=1, fontsize=10,
                        loc='outside center right', labelspacing=0.4,
                        handlelength=1.5, handletextpad=0.5, borderpad=0.4)

    plt.yticks(np.arange(0, 1.1, 0.1))

    for axes in p.axes.flat:
        axes.yaxis.set_major_formatter(FuncFormatter(comma_formatter))

    output_path = f'plots/{metric_name}.pdf'
    plt.savefig(output_path)  # sin bbox_inches='tight': constrained_layout ya calcula el encuadre
    plt.close()


for metric in METRICS:
    plot_metric(full_dataframe, metric)