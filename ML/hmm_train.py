import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from hmmlearn.hmm import GaussianHMM  
from sklearn.preprocessing import StandardScaler
import joblib
import torch 

from data_manipulations import prepare_data_ratio

def directional_accuracy_score(y_true:pd.Series, y_pred_proba):
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true.to_numpy(), dtype=torch.float32)
    if not isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = torch.tensor(y_pred_proba, dtype=torch.float32)
    y_pred = (y_pred_proba > 0.5).float()

    sign_agreement = torch.sign(y_pred - 1) * torch.sign(y_true - 1)

    confidence_weight = torch.abs(y_pred_proba - 0.5) * 2 

    return torch.mean(sign_agreement * confidence_weight).item()

def main(data_read_path: str, data_write_path: str, model_save_path: str):
    df = pd.read_csv(data_read_path).drop('Time', axis='columns')
    #df = yf.Ticker('TSLA').history(interval="1h", start="2015-01-01")
    print(df.head())
    df.to_csv(data_write_path)

    df['next_ratio'] = df['Close'].shift(-1) / df['Close']

    df["RSI"] = ta.rsi(close=df["Close"], length=24)
    df["Stochastic"] = ta.stoch(high=df["High"], low=df["Low"], close=df["Close"])["STOCHk_14_3_3"]
    df["MACD"] = ta.macd(close=df["Close"])["MACD_12_26_9"]
    df["Cycle"] = ta.ebsw(close=df["Close"], length=30)  

    df = df.dropna()
    train_part = 0.8
    train_size = int(len(df) * train_part)

    df_train = df[:train_size]
    df_test = df[train_size:]

    X_train = df_train.drop(["next_ratio"], axis=1).values
    y_train = (df_train["next_ratio"] > 1).astype(int)

    X_test = df_test.drop(["next_ratio"], axis=1).values
    y_test = (df_test["next_ratio"] > 1).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df['y'] = pd.concat([y_train, y_test])

    hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=500, random_state=42)
    hmm_model.fit(X_train_scaled)

    df_train["Predicted_State"] = hmm_model.predict(X_train_scaled)
    df_train["Predicted_Change"] = (hmm_model.predict_proba(X_train_scaled)[:, 1] > 0.5).astype(int)

    accuracy = (df_train["Predicted_Change"] == y_train).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    joblib.dump(hmm_model, model_save_path)

    y_pred_proba = hmm_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(directional_accuracy_score(y_true=y_test, y_pred_proba=y_pred))

if __name__ == '__main__':
    main(
        data_read_path=r"/home/alex/BitcoinScalper/data_collecting/tinkoff_data/prices_massive_LKOH_4_HOUR_2025-01-25.csv",
        data_write_path=r"/home/alex/BitcoinScalper/dataframes/LKOH_hmm.csv",
        model_save_path=r"/home/alex/BitcoinScalper/ML/models/HMM_LKOH.pkl",
    )