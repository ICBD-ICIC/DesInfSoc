import pandas as pd

COMMUNICATION_STYLE = pd.read_csv('../outputs/user/communication_style.csv')
PERSONALITY = pd.read_csv('../outputs/user/personality.csv')
PERSONALITY_TRAITS = pd.read_csv('../outputs/user/personality_traits.csv')
OUTPUT_FILE = '../outputs/user/user_features.csv'

BIG_FIVE_CATEGORIES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
COMMUNICATION_STYLE_CATEGORIES = ['action-seeking', 'fact-oriented', 'information-seeking', 'self-revealing']

def discretize_big_five(user):
    bstring_category = "".join(list(map(lambda category: '1' if user[category] > 0.5 else '0', BIG_FIVE_CATEGORIES)))
    return int(bstring_category, 2)


def discretize_psychographics(user):
    bstring_category = "".join(list(map(lambda category: '1' if user[category] > 0.5 else '0', COMMUNICATION_STYLE_CATEGORIES)))
    bstring_category += '1' if user['rational'] > 0.5 else '0'
    return int(bstring_category, 2)


users = pd.merge(COMMUNICATION_STYLE, PERSONALITY, on='id', how='outer')
users = pd.merge(PERSONALITY_TRAITS, users, on='id', how='outer')

users_features = []

for index, user in users.iterrows():
    user_big_five = discretize_big_five(user)
    user_physiographic = discretize_psychographics(user)
    users_features.append({'username': user['id'], 'big_five': user_big_five, 'psychographics': user_physiographic})

users_features_df = pd.DataFrame(users_features)
users_features_df.to_csv(OUTPUT_FILE, index=False)
