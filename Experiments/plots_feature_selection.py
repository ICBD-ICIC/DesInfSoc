import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from matplotlib.patches import Patch

RESULTS_FOLDER_PATH = 'binary/feature-selection/results-all(spread20,balanced)/'
METRIC = 'precision'


HATCHES = ['\\\\', '-', '//', '..', '', 'oo', '++', '||', 'XX', 'OO', '\\', '--', '/', '.', '+', 'o', '|', 'X', 'o']


def paper_experiment_one(dataframe):
    features = {'emotion': 'Only emotion',
                'emotion+linguistic-ratio': 'Emotion + Linguistic Ratio',
                'personality': 'Only personality',
                'linguistic-ratio+personality': 'Personality + Linguistic Ratio',
                'sentiment': 'Only sentiment',
                'linguistic-ratio+sentiment': 'Sentiment + Linguistic Ratio',
                'emotion+personality': 'Emotion + Personality',
                'emotion+linguistic-ratio+personality': 'Emotion + Personality + Linguistic Ratio',
                'emotion+sentiment': 'Emotion + Sentiment',
                'emotion+linguistic-ratio+sentiment': 'Emotion + Sentiment + Linguistic Ratio',
                'personality+sentiment': 'Personality + Sentiment',
                'linguistic-ratio+personality+sentiment': 'Personality + Sentiment + Linguistic Ratio',
                'emotion+sentiment+personality': 'Emotion + Sentiment + Personality',
                'emotion+linguistic-ratio+sentiment+personality': 'Emotion + Sentiment + Personality + Linguistic Ratio',
                }
    df = dataframe.loc[dataframe['feature'].isin(features.keys())]
    df['feature'] = pd.Categorical(df['feature'], categories=features.keys(), ordered=True)
    df['feature'] = df['feature'].replace(features)

    predictions = {'predominant_emotion': 'P5',
                   'predominant_sentiment': 'P4',
                   'valence_ratio': 'P6',
                   'abusive_ratio_interval': 'P1',
                   'mfd_ratio': 'P3',
                   'polarization_ratio_interval': 'P2'}
    df['prediction'] = df['prediction'].replace(predictions)

    df = df.sort_values(by=['feature', 'prediction'])
    title = 'Precision: End to end, Spread = 20'
    return df, title


folders_list = os.listdir(RESULTS_FOLDER_PATH)

plot_data = []

for folder in folders_list:
    file_list = os.listdir(RESULTS_FOLDER_PATH + folder)
    for file in file_list:
        metrics = open(RESULTS_FOLDER_PATH + folder + "/" + file, "r").read()
        metrics = " ".join(metrics.split())
        metrics = metrics.split('}{', 1)[0]
        metrics = metrics + '}' if metrics[-1] != '}' else metrics
        metrics = json.loads(metrics)
        spread = file.split(',')[0].split('_')[1].replace('SPREAD','')
        model = file.split(',')[1]
        prediction: str = file.split(',')[2]
        feature = folder.replace('experiments-', '')
        for name, value in metrics.items():
            model_data = {'metric_name': name, 'metric_value': value, 'model': model,
                          'prediction': prediction, 'spread': spread, 'feature': feature}
            plot_data.append(model_data)

dataframe = pd.DataFrame(plot_data)
dataframe, title = paper_experiment_one(dataframe)
dataframe = dataframe.loc[dataframe['metric_name'] == METRIC]

sns.set_style("whitegrid")
sns.set_palette("pastel")

p = sns.catplot(data=dataframe, x="prediction", y="metric_value", hue="feature", col='spread', kind="bar",
                edgecolor='black', legend=False)
p.set(xlabel=None, ylabel=None)
p.set_titles("{col_name}")

colors = []
features = dataframe['feature'].unique()

# Loop through the bars and assign hatches
for axes in p.axes.flat:
    axes.set_title(axes.get_title().replace('_', ' ').capitalize())
    for bar in axes.patches:
        bar_color = bar.get_facecolor()
        colors.append(bar_color) if bar_color not in colors else colors
        bar.set_hatch(HATCHES[colors.index(bar_color)])
    # shows label with value of each bar
    # for i in axes.containers:
    #     axes.bar_label(i, )

legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=HATCHES[i], label=features[i])
                  for i in range(0, len(colors))]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=10)
plt.subplots_adjust(right=0.6)

plt.title(title)

plt.show()
