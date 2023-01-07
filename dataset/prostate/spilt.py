import pandas as pd
import numpy as np

df = pd.read_csv(r'dataset\prostate\ucl.csv')
num = df.shape[0]
train_num = int(num*0.8)
idx = np.random.permutation(num)
train_idx = idx[:train_num]
test_idx = idx[train_num:]

train_csv = df.iloc[train_idx]
test_csv = df.iloc[test_idx]
train_csv.to_csv(r'dataset\prostate\ucltrain.csv', index=None)
test_csv.to_csv(r'dataset\prostate\ucltest.csv', index=None)