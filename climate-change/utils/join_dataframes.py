# # Join valence positive and negative into a single file
# # Join mfd virtue and vice into a single file
#
# import pandas as pd
#
# VALENCE_NEGATIVE = pd.read_csv('../outputs/valence_negative.csv')
# VALENCE_POSITIVE = pd.read_csv('../outputs/valence_positive.csv')
# MFD_VIRTUE = pd.read_csv('../outputs/mfd_virtue.csv')
# MFD_VICE = pd.read_csv('../outputs/mfd_vice.csv')
#
# valence = pd.merge(VALENCE_NEGATIVE, VALENCE_POSITIVE, on='id', how='outer')
# valence.to_csv('../outputs/valence.csv')
#
# mfd = pd.merge(MFD_VICE, MFD_VIRTUE, on='id', how='outer')
# mfd.to_csv('../outputs/mfd.csv')

# Join user features into a single file

import pandas as pd

COMMUNICATION_STYLE = (pd.read_csv('../outputs/user/communication_style.csv')
                       .rename(columns=lambda col: 'communication_style@' + col if col != 'id' else col))
PERSONALITY = (pd.read_csv('../outputs/user/personality.csv')
               .rename(columns=lambda col: 'personality@' + col if col != 'id' else col))
PERSONALITY_TRAITS = (pd.read_csv('../outputs/user/personality_traits.csv')
                      .rename(columns=lambda col: 'personality_traits@' + col if col != 'id' else col))
OUTPUT_FILE = '../outputs/user/users_features_llm.csv'

features = pd.merge(COMMUNICATION_STYLE, PERSONALITY, on='id', how='outer')
features = pd.merge(features, PERSONALITY_TRAITS, on='id', how='outer')

features.to_csv(OUTPUT_FILE, index=False)