import pandas as pd

###########################################################
#### GET NETWORK USERS WITH AT LEAST 20 TWEETS NO RTs #####
###########################################################

# NETWORK_USERS_TWEETS_FILE = 'dataset/all-languages/network_users_tweets.csv'
NETWORK_USERS_BY_ACTIVITY = '../dataset/intermediate/network_users_by_activity.csv'
# NETWORK_USERS_TWEETS_BY_ACTIVITY = 'outputs/network_users_tweets_by_activity.csv'

# network_users_tweets = pd.read_csv(NETWORK_USERS_TWEETS_FILE)
network_users_by_activity = pd.read_csv(NETWORK_USERS_BY_ACTIVITY)
network_users_by_activity = network_users_by_activity.drop(network_users_by_activity[network_users_by_activity['tweet_amount'] < 20].index)

network_users_by_activity.to_csv('../dataset/network_active_users.csv')

#
# network_users_tweets_by_activity = network_users_by_activity.set_index('id').join(network_users_tweets.set_index('id'),
#                                                                                   how='inner')
# network_users_tweets_by_activity = network_users_tweets_by_activity.astype({'tweet_amount': int})
# network_users_tweets_by_activity.index = network_users_tweets_by_activity.index.astype(int)
#
# network_users_tweets_by_activity.to_csv(NETWORK_USERS_TWEETS_BY_ACTIVITY)

#########################################################################
#### FILTER BIG FIVE RESULTS - USERS WITH AT LEAST 20 TWEETS NO RTs #####
#########################################################################

# ALL_BIG_FIVE = 'dataset/personality/big_five_clean.csv'
# BIG_FIVE_ACTIVE_USERS = 'outputs/big_five_active_users.csv'
#
# all_big_five = pd.read_csv(ALL_BIG_FIVE)
#
# big_five_active_users = all_big_five.loc[all_big_five['id'].isin(network_users_by_activity['id']),:]
# print(big_five_active_users)
#
# big_five_active_users.to_csv(BIG_FIVE_ACTIVE_USERS, index=False)

#################################################################################
#### GET REMAINING USERS TO ANALYZE AND FILTER RESULTS - PERSONALITY TRAITS #####
#################################################################################

# OLD_PERSONALITY_TRAIT = 'outputs/personality-old/symanto_personality_traits_1703689056.8867393.csv'
# NEW_PERSONALITY_TRAIT = 'outputs/personality/symanto_personality_traits.csv'
# NETWORK_USERS_TWEETS_TO_ANALYZE = 'dataset/network_users_tweets_to_analyze_personality_traits.csv'
#
# personality_trait = pd.read_csv(OLD_PERSONALITY_TRAIT)
# personality_trait = personality_trait.loc[personality_trait['id'].isin(network_users_by_activity['id']), :]
# personality_trait.to_csv(NEW_PERSONALITY_TRAIT, index=False)
#
# personality_trait = pd.read_csv(NEW_PERSONALITY_TRAIT)
# all_users = pd.read_csv(NETWORK_USERS_TWEETS_BY_ACTIVITY)
# network_user_tweets_to_analyze = all_users.loc[~all_users['id'].isin(personality_trait['id']), :]
# network_user_tweets_to_analyze.to_csv(NETWORK_USERS_TWEETS_TO_ANALYZE)

#################################################################################
#### GET REMAINING USERS TO ANALYZE AND FILTER RESULTS - COMMUNICATION STYLE ####
#################################################################################

# OLD_COMMUNICATION_STYLE = 'outputs/personality-old/symanto_communication_style_1703782782.6915822.csv'
# NEW_COMMUNICATION_STYLE = 'outputs/personality/symanto_communication_style.csv'
# NETWORK_USERS_TWEETS_TO_ANALYZE = 'dataset/network_users_tweets_to_analyze_communication_style.csv'
#
# communication_style = pd.read_csv(OLD_COMMUNICATION_STYLE)
# communication_style = communication_style.loc[communication_style['id'].isin(network_users_by_activity['id']), :]
# communication_style.to_csv(NEW_COMMUNICATION_STYLE, index=False)
#
# communication_style = pd.read_csv(NEW_COMMUNICATION_STYLE)
# all_users = pd.read_csv(NETWORK_USERS_TWEETS_BY_ACTIVITY)
# network_user_tweets_to_analyze = all_users.loc[~all_users['id'].isin(communication_style['id']), :]
# network_user_tweets_to_analyze.to_csv(NETWORK_USERS_TWEETS_TO_ANALYZE)
