import pandas as pd
import torch
import numpy as np
#from copy import deepcopy as dc

# Optimized: Combined normalization and reshaping to reduce redundant calculations.
def split_data(df: pd.DataFrame, train_part: float, scaler=None):
    train_size = int(len(df) * train_part)

    next_ratio_cols = [col for col in df.columns if "next_ratio" in col or "final_direction" in col]
    X = df.drop(next_ratio_cols, axis=1)  
    y = df[next_ratio_cols]

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    if scaler:
        return (
            scaler.fit_transform(X_train.to_numpy(dtype=np.float32)), scaler.transform(X_test.to_numpy(dtype=np.float32)),
            scaler.fit_transform(y_train.to_numpy(dtype=np.float32).reshape(-1, 1)), scaler.transform(y_test.to_numpy(dtype=np.float32).reshape(-1, 1)),
        )
    else:
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
# Optimized: Removed unnecessary `unsqueeze(1)` calls for reshaping.
def create_tensors(X_train, X_test, y_train, y_test):
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

def calculate_n_next_ratio(df, n):
    for i in range(-1, (-1*n)-1, -1):
        df[f'{i}_next_ratio'] = df['Close'].shift(i-1) / df['Close'].shift(i)
    return df

def calculate_n_prev_ratio(df, n):
    for i in range(1, n + 1):
        df[f'{i}_prev_ratio'] = df['Close'].shift(i) / df['Close'].shift(i + 1)
    return df

def prepare_data_ratio(df: pd.DataFrame, data_write_path:str = '', n_prev_ratio=7, n_next_ratio=5, window_size=30):
    try:
        df = df.drop(['Date'], axis=1)
    except Exception as e:
        print(e)

    df = calculate_n_prev_ratio(df, n_prev_ratio)
    df = calculate_n_next_ratio(df, n_next_ratio)
    df[f'{window_size}_prev_ratio_mean'] = df['1_prev_ratio'].rolling(window=window_size).mean()
    #df['final_direction'] = 

    df = df.dropna(axis='rows')
    if data_write_path != '':
        df.to_csv(data_write_path)

    print(len(df))
    print(df.columns)
    return df