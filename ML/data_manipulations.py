import pandas as pd
import torch
#from copy import deepcopy as dc

# Optimized: Combined normalization and reshaping to reduce redundant calculations.
def split_data(df: pd.DataFrame, train_part: float):
    train_size = int(len(df) * train_part)
    X = df.drop('next_ratio', axis=1)  
    y = df['next_ratio']

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

# Optimized: Removed unnecessary `unsqueeze(1)` calls for reshaping.
def create_tensors(X_train, X_test, y_train, y_test):
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

def _n_ratio(df, n):
    for i in range(1, n + 1):
        df[f'{i}_prev_ratio'] = df['Close'].shift(i) / df['Close'].shift(i + 1)
    return df

def prepare_data_ratio(df: pd.DataFrame, data_write_path:str = '', n_ratio=7, window_size=30):
    try:
        df = df.drop(['Date'], axis=1)
    except Exception as e:
        print(e)

    df['next_ratio'] = df['Close'].shift(-1) / df['Close']
    df = _n_ratio(df, n_ratio)
    df[f'{window_size}_prev_ratio_mean'] = df['1_prev_ratio'].rolling(window=window_size).mean()

    df = df.dropna()
    if data_write_path != '':
        df.to_csv(data_write_path)

    print(len(df))
    print(df.columns)
    return df