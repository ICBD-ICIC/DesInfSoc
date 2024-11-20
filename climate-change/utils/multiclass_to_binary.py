import pandas as pd

original_path = '../dataset/CONTEXT_LLM_distance.csv'
context = pd.read_csv(original_path)

# 0: [0,0.50)
# 1: [0.50,1]
columns_to_modify = ['abusive_amount_interval_gt', 'abusive_ratio_interval_gt', 'polarization_amount_interval_gt', 'polarization_ratio_interval_gt']
context[columns_to_modify] = context[columns_to_modify].replace(1, 0)
context[columns_to_modify] = context[columns_to_modify].replace(2, 1)
context[columns_to_modify] = context[columns_to_modify].replace(3, 1)

# '39': 'predominant_emotion'
# 0: neutral, joy, surprise
# 1: fear, anger, disgust, sadness
context['predominant_emotion_gt'] = context['predominant_emotion_gt'].replace(4, 0)
context['predominant_emotion_gt'] = context['predominant_emotion_gt'].replace(6, 0)
context['predominant_emotion_gt'] = context['predominant_emotion_gt'].replace(3, 1)
context['predominant_emotion_gt'] = context['predominant_emotion_gt'].replace(2, 1)
context['predominant_emotion_gt'] = context['predominant_emotion_gt'].replace(5, 1)

# 0: virtue >= vice
# 1: virtue < vice
columns_to_modify = ['mfd_amount_gt', 'mfd_ratio_gt']
context[columns_to_modify] = context[columns_to_modify].replace(1, 0)
context[columns_to_modify] = context[columns_to_modify].replace(2, 1)

# 0: positive >= negative
# 1: positive < negative
columns_to_modify = ['valence_amount_gt', 'valence_ratio_gt']
context[columns_to_modify] = context[columns_to_modify].replace(1, 0)
context[columns_to_modify] = context[columns_to_modify].replace(2, 1)

# 0: positive, neutral
# 1: negative
context['predominant_sentiment_gt'] = context['predominant_sentiment_gt'].replace(1, 0)
context['predominant_sentiment_gt'] = context['predominant_sentiment_gt'].replace(2, 1)

context.to_csv(original_path.replace('.csv', '-binary.csv'), index=False)
