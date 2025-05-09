import pandas as pd
import numpy as np

df = pd.read_csv('winequality-red.csv')

np.random.seed(42)

missing_fraction_1 = 0.1

missing_fraction_2 = 0.15

features_to_nan = np.random.choice(df.columns[:-1], size=2, replace=False)

print(features_to_nan)

n_missing = int(len(df) * missing_fraction_1)
missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
df.loc[missing_indices, features_to_nan[0]] = np.nan

n_missing = int(len(df) * missing_fraction_2)
missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
df.loc[missing_indices, features_to_nan[1]] = np.nan

df.to_csv('winequality-red-new.csv', index=False)