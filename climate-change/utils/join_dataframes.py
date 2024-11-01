# Join valence positive and negative into a single file
# Join mfd virtue and vice into a single file

import pandas as pd

VALENCE_NEGATIVE = pd.read_csv('../outputs/valence_negative.csv')
VALENCE_POSITIVE = pd.read_csv('../outputs/valence_positive.csv')
MFD_VIRTUE = pd.read_csv('../outputs/mfd_virtue.csv')
MFD_VICE = pd.read_csv('../outputs/mfd_vice.csv')

valence = pd.merge(VALENCE_NEGATIVE, VALENCE_POSITIVE, on='id', how='outer')
valence.to_csv('../outputs/valence.csv')

mfd = pd.merge(MFD_VICE, MFD_VIRTUE, on='id', how='outer')
mfd.to_csv('../outputs/mfd.csv')
