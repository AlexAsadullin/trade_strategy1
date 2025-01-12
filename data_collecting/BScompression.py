import pandas as pd
from datetime import datetime as dt
def n_prev_ratio(df: pd.DataFrame, n: int):
    for i in range(1, n):
        df[f'{i+1}_prev_ratio'] = df['price'].shift(i) / df['price'].shift(i+1)
    return df

def aggregate_trades(df, max_rows=100):
    result = []
    i = 0
    while i < len(df):
        j = i
        begin_deal = df.loc[i]
        current_side = begin_deal['side']
        current_group = []
        while j < len(df) and df.loc[j, 'side'] == current_side:
            current_group.append(df.loc[j])
            j += 1
        if current_group:
            aggregated_row = pd.DataFrame(current_group).agg({'price': 'mean', 'size': 'sum'})
            aggregated_row['side'] = current_side
            result.append(aggregated_row)
        i = j + 1
    return pd.DataFrame(result)


df = pd.read_csv(rf'/home/alex/BitcoinScalper/dataframes/dataset_train_bull_{dt.now().strftime}.csv', index_col=0)
df['side'] = df['side'].map({'Buy': 1, 'Sell': -1})
df = df.drop(['symbol'], axis=1)
df = df.drop(['time'], axis=1)

df = aggregate_trades(df)
df.to_csv(r'/home/alex/BitcoinScalper/dataframes/aggregated_deals.csv')

df = n_prev_ratio(df, 15)
df['next_ratio'] = df['price'].shift(-1) / df['price']
df['prev_ratio'] = df['price'] / df['price'].shift(1)
df['prev_ratio_mean'] = df['prev_ratio'].rolling(100).mean()
df['side'] = df['side'].map({'Buy': 1, 'Sell': -1})

now = dt.now().strftime("%Y-%m-%d")

df.dropna(inplace=True)
df.to_csv(rf'/home/alex/BitcoinScalper/dataframes/compressed_dataset_{now}.csv')

import gzip
import shutil

with open(rf'/home/alex/BitcoinScalper/dataframes/compressed_dataset_{now}.csv', 'rb') as f_in:
    with gzip.open(rf'/home/alex/BitcoinScalper/dataframes/compressed_dataset_{now}.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)