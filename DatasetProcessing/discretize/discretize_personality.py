import pandas as pd

NETWORK_USERS_BY_ACTIVITY = '../dataset/network_users_by_activity.csv'
PERSONALITY_TRAITS = '../dataset/personality/personality_traits_active_network_users.csv'
COMMUNICATION_STYLE = '../dataset/personality/communication_style_active_network_users.csv'
BIG_FIVE = '../dataset/personality/big_five_active_network_users.csv'

OUTPUT_FILE = '../outputs/discretized_personality.csv'

BIG_FIVE_CATEGORIES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
COMMUNICATION_STYLE_CATEGORIES = ['action-seeking', 'fact-oriented', 'information-seeking', 'self-revealing']


def discretize_big_five(big_five_user):
    bstring_category = "".join(list(map(lambda category: '1' if big_five_user[category] > 0.5 else '0', BIG_FIVE_CATEGORIES)))
    return int(bstring_category, 2)


def discretize_psychographics(user_communication_style, user_personality_traits):
    bstring_category = "".join(list(map(lambda category: '1' if user_communication_style[category] > 0.5 else '0', COMMUNICATION_STYLE_CATEGORIES)))
    bstring_category += '1' if user_personality_traits['rational'] > 0.5 else '0'
    return int(bstring_category, 2)


users = pd.read_csv(NETWORK_USERS_BY_ACTIVITY).set_index('id')
users = users.drop(users[users['tweet_amount'] < 20].index)

personality_traits = pd.read_csv(PERSONALITY_TRAITS).set_index('id')
communication_style = pd.read_csv(COMMUNICATION_STYLE).set_index('id')
big_five = pd.read_csv(BIG_FIVE).set_index('id')

discretized_users = pd.DataFrame(columns=['user_id', 'big_five', 'symanto_psychographics'])

for user_id in users.index:
    discretized_user = {'user_id': user_id,
                        'big_five': discretize_big_five(big_five.loc[user_id]),
                        'symanto_psychographics': discretize_psychographics(communication_style.loc[user_id],
                                                                            personality_traits.loc[user_id])}
    discretized_users.loc[len(discretized_users)] = discretized_user

discretized_users.to_csv(OUTPUT_FILE, index=False)
