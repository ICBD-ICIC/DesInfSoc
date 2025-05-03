import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
import numpy as np
from matplotlib import gridspec

EXPERIMENT_TYPES = ['distance', 'pattern_matching']
METRICS = ['precision', 'recall', 'f1']

label_prediction_map = {
    "abusive_ratio_interval_gt": "P1",
    "polarization_ratio_interval_gt": "P2",
    "mfd_ratio_gt": "P3",
    "predominant_sentiment_gt": "P4",
    "predominant_emotion_gt": "P5",
    "valence_ratio_gt": "P6"
}

sns.set_style("whitegrid")
sns.set_palette("colorblind")  # Colorblind-friendly palette

for METRIC in METRICS:
    fig = plt.figure(figsize=(12, 4))  # << reduced width here
    spec = gridspec.GridSpec(
        ncols=len(EXPERIMENT_TYPES),
        nrows=1,
        width_ratios=[1] * len(EXPERIMENT_TYPES),
        wspace=0  # << No space between inner plots
    )
    axes = [fig.add_subplot(spec[i]) for i in range(len(EXPERIMENT_TYPES))]

    all_handles = []
    all_labels = []

    for idx, EXPERIMENT_TYPE in enumerate(EXPERIMENT_TYPES):
        FOLDER_PATHS = [f'../experiments/experiments-{EXPERIMENT_TYPE}/']
        file_list = []

        for folder_path in FOLDER_PATHS:
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                file_list.append(file_path)

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
                model_data = {
                    'metric_name': name,
                    'metric_value': value,
                    'model': model.replace('_', ' ').title(),
                    'prediction': label_prediction_map[prediction]
                }
                plot_data.append(model_data)

        dataframe = pd.DataFrame(plot_data)
        dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

        sns.barplot(
            data=dataframe,
            x="prediction",
            y="metric_value",
            hue="model",
            ax=axes[idx],
            edgecolor='black'
        )

        axes[idx].set_title(f'{EXPERIMENT_TYPE.replace("_", " ").title()}', fontsize=20)
        axes[idx].tick_params(axis='x', labelsize=11)
        axes[idx].set_xlabel("")
        if idx == 0:
            axes[idx].set_ylabel(METRIC.capitalize(), fontsize=20)
            axes[idx].set_yticks(np.arange(0, 1.1, 0.1))
        else:
            axes[idx].set_ylabel("")
            axes[idx].set_yticks(np.arange(0, 1.1, 0.1))  # Keep tick locations for gridlines
            axes[idx].set_yticklabels([])  # Hide tick numbers
            axes[idx].tick_params(axis='y', left=False)  # Hide tick marks (lines)

        axes[idx].tick_params(labelsize=20)

        if idx == 0:
            handles, labels = axes[idx].get_legend_handles_labels()
            all_handles = handles
            all_labels = labels

        axes[idx].get_legend().remove()

    # Shared legend on the right
    fig.legend(
        all_handles,
        all_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=20,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to leave space for legend

    plt.subplots_adjust(wspace=0.4)  # Adjust spacing here
    plt.savefig(f'{METRIC}.pdf', bbox_inches='tight')
    #plt.show()
