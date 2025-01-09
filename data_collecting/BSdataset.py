import os

import pandas as pd
import json
from datetime import datetime as dt

cols = ['execId',
        'symbol',
        'price',
        'size',
        'side',
        'time',
        'isBlockTrade',
        ]

df = pd.DataFrame([], columns=cols)
for i in os.listdir(r'/home/alex/BitcoinScalper/data_collecting/training_data'):
    with open(rf'/home/alex/BitcoinScalper/data_collecting/training_data/{i}') as f:
        data = json.load(f)
        new_df = pd.DataFrame(data, columns=cols)
    df = pd.concat([df, new_df], axis=0)
print(df.head())
df = df.drop(['execId', 'isBlockTrade', 'time', 'symbol'], axis='columns')
df['side'] = df['side'].map({'Buy': 1, 'Sell': -1})

df.to_csv(rf'/home/alex/BitcoinScalper/dataframes/my_dataset_{dt.now().strftime('%Y-%m-%d')}.csv')

for i in os.listdir('/home/alex/BitcoinScalper/data_collecting/training_data'):
    os.remove(f'//home/alex/BitcoinScalper/data_collecting/training_data/{i}')