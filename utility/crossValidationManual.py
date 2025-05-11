import numpy as np
from sklearn.model_selection import StratifiedKFold

data = wine_data.to_numpy()

def split_data_with_cv(data, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    
    test_size = int(test_ratio * len(data))
    test_indices = indices[:test_size]
    trainval_indices = indices[test_size:]

    test = data[test_indices]
    trainval = data[trainval_indices]

    return trainval, test

def replace_nans(*datasets):
    train = datasets[0]
    train_mean = np.nanmean(train[:, :-1], axis=0)
    for dataset in datasets:
        for i in range(dataset.shape[1] - 1):
            nan_mask = np.isnan(dataset[:, i])
            dataset[nan_mask, i] = train_mean[i]
    return datasets

trainval, test = split_data_with_cv(data, test_ratio=0.1)

trainval, test = replace_nans(trainval, test)

X_trainval, y_trainval = trainval[:, :-1], trainval[:, -1].astype(int)
X_test, y_test = test[:, :-1], test[:, -1].astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]