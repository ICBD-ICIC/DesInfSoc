import pandas as pd

# NETWORK_USERS_BY_ACTIVITY = pd.read_csv('dataset/network_users_by_activity.csv').set_index('id')
PERSONALITY_TRAITS = pd.read_csv('dataset/personality/personality_traits_active_network_users.csv').set_index('id')
COMMUNICATION_STYLE = pd.read_csv('dataset/personality/communication_style_active_network_users.csv').set_index('id')
BIG_FIVE = pd.read_csv('dataset/personality/big_five_active_network_users.csv').set_index('id')

# OUTPUT_FILE = '../outputs/discretized_personality.csv'

BIG_FIVE_CATEGORIES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
COMMUNICATION_STYLE_CATEGORIES = ['action-seeking', 'fact-oriented', 'information-seeking', 'self-revealing']


def discretize_big_five(user_id):
    big_five_user = BIG_FIVE.loc[user_id]
    bstring_category = "".join(list(map(lambda category: '1' if big_five_user[category] > 0.5 else '0', BIG_FIVE_CATEGORIES)))
    return {'big_five': int(bstring_category, 2)}


def discretize_psychographics(user_id):
    user_communication_style = COMMUNICATION_STYLE.loc[user_id]
    user_personality_traits = PERSONALITY_TRAITS.loc[user_id]
    bstring_category = "".join(list(map(lambda category: '1' if user_communication_style[category] > 0.5 else '0', COMMUNICATION_STYLE_CATEGORIES)))
    bstring_category += '1' if user_personality_traits['rational'] > 0.5 else '0'
    return {'psychographics': int(bstring_category, 2)}


# discretized_users = pd.DataFrame(columns=['user_id', 'big_five', 'symanto_psychographics'])

# for user_id in users.index:
#     discretized_user = {'user_id': user_id,
#                         'big_five': discretize_big_five(user_id),
#                         'symanto_psychographics': discretize_psychographics(user_id,
#                                                                             user_id)}
#     discretized_users.loc[len(discretized_users)] = discretized_user
#
# discretized_users.to_csv(OUTPUT_FILE, index=False)
