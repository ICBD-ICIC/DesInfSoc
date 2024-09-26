BIG_FIVE_CATEGORIES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
COMMUNICATION_STYLE_CATEGORIES = ['action-seeking', 'fact-oriented', 'information-seeking', 'self-revealing']


def discretize_big_five(user):
    bstring_category = "".join(list(map(lambda category: '1' if user[category] > 0.5 else '0', BIG_FIVE_CATEGORIES)))
    return {'big_five': int(bstring_category, 2)}


def discretize_psychographics(user):
    bstring_category = "".join(list(map(lambda category: '1' if user[category] > 0.5 else '0', COMMUNICATION_STYLE_CATEGORIES)))
    bstring_category += '1' if user['rational'] > 0.5 else '0'
    return {'psychographics': int(bstring_category, 2)}
