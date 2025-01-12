import pandas as pd
import numpy as np
import plotly.graph_objects as go



def find_min_max(df: pd.DataFrame, window_size=10):
    df["Max"] = df["Close"].rolling(window=2*window_size+1, center=True).max()
    df["Min"] = df["Close"].rolling(window=2*window_size+1, center=True).min()

    df["Max"] = df["Max"].fillna(df["Close"].expanding(1).max())
    df["Min"] = df["Min"].fillna(df["Close"].expanding(1).min())
    
    df.loc[df.index > int(len(df) - window_size*1.05), ['Max', 'Min']] = np.nan

    df['Max'] = df['Max'].ffill()
    df['Min'] = df['Min'].ffill()
    return df

def find_entry_points(df: pd.DataFrame, window_size=10):
    shifted_full = df['Max'].shift(window_size * -1)
    shifted_08 = df['Max'].shift(int(window_size * -1 * 0.8))
    shifted_06 = df['Max'].shift(int(window_size * -1 * 0.6))
    shifted_06 = df['Max'].shift(int(window_size * -1 * 0.4))

    condition1 = shifted_full > df['Max']
    condition2 = shifted_08 > df['Max']
    condition3 = shifted_06 > df['Max']

    df['IsEntryPoint'] = (condition1 & condition2 & condition3).astype(int)
    return df

def plot_minmax_isentrypoint(df: pd.DataFrame, chart_path: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode='lines',
        line=dict(color="blue"), name='Close'))

    entry_points = df[df['IsEntryPoint'] == 1]
    fig.add_trace(go.Scatter(x=entry_points.index,
        y=entry_points['Close'], mode='markers', marker=dict(color='yellow', size=4), name='Entry Points'))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Max"], mode='lines', marker=dict(color='green', size=0.8), name='Max Points'))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Min"], mode='lines', marker=dict(color='red', size=0.8), name='Min Points'))
    fig.write_html(chart_path)
    return fig


if __name__ == '__main__':
    df = pd.read_csv(r'/home/alex/BitcoinScalper/data_collecting/tinkoff_data/prices_massive_SBER_4_HOUR_2025-01-08.csv', index_col=0)
    WINDOW_SIZE = 100

    df = find_min_max(df=df, window_size=WINDOW_SIZE)
    df = find_entry_points(df=df, window_size=WINDOW_SIZE)
    fig = plot_minmax_isentrypoint(df=df, chart_path=r'/home/alex/BitcoinScalper/charts/MinMaxEntryPoint.html')
    
    df.to_csv(r'/home/alex/BitcoinScalper/dataframes/full_data.csv')
    print(df.head())
    print(len(df[df['IsEntryPoint'] == 0]), len(df))