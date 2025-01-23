import numpy as np
import pandas as pd
import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM  
import joblib

from data_manipulations import prepare_data_ratio

def main(data_read_path: str, data_write_path: str, model_save_path: str):
    data = pd.read_csv(data_read_path)
    df = prepare_data_ratio(data, data_write_path=data_write_path)

    df["RSI"] = ta.rsi(close=df["Close"], length=14)
    df["Stochastic"] = ta.stoch(high=df["High"], low=df["Low"], close=df["Close"])["STOCHk_14_3_3"]
    df["MACD"] = ta.macd(close=df["Close"])["MACD_12_26_9"]
    df["Cycle"] = ta.ebsw(close=df["Close"], length=20)  

    df = df.dropna()
    train_part = 0.8
    train_size = int(len(df) * train_part)

    df_train = df[:train_size]
    df_test = df[train_size:]

    X_train = df_train.drop(["next_ratio"], axis=1).values
    y_train = (df_train["next_ratio"] > 1).astype(int)

    X_test = df_test.drop(["next_ratio"], axis=1).values
    y_test = (df_test["next_ratio"] > 1).astype(int)

    hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    hmm_model.fit(X_train)

    df_train["Predicted_State"] = hmm_model.predict(X_train)
    df_train["Predicted_Change"] = (hmm_model.predict_proba(X_train)[:, 1] > 0.5).astype(int)

    accuracy = (df_train["Predicted_Change"] == y_train).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    joblib.dump(hmm_model, model_save_path)
    print(f"HMM model saved as '{model_save_path}'")

if __name__ == '__main__':
    main(
        data_read_path=r"/home/alex/BitcoinScalper/dataframes/TSLA.csv",
        data_write_path=r"/home/alex/BitcoinScalper/dataframes/TSLA_hmm.csv",
        model_save_path=r"/home/alex/BitcoinScalper/ML/models/HMM_TSLA.pkl",
    )