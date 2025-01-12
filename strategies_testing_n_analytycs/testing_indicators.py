import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly._subplots import make_subplots

df = pd.read_csv('/home/alex/BitcoinScalper/dataframes/full_data.csv', index_col=0)

xserie = df["Open"]  # Дополнительная серия
xserie_a = df["High"]  # Первая дополнительная серия
xserie_b = df["Low"]   # Вторая дополнительная серия

df['RSI'] = ta.rsi(
    close=df["Close"],         # Основная временная серия
    length=14,                 # Длина окна RSI
    scalar=100,                # Масштабирование (обычно 100 для RSI)
    drift=1,                   # Шаг для diff
    offset=0,                  # Без смещения
    #signal_indicators=True,    # Включение сигналов
    xa=80,                     # Верхний порог RSI
    xb=20,                     # Нижний порог RSI
    xserie=xserie,             # Дополнительная временная серия
    xserie_a=xserie_a,         # Первая временная серия
    xserie_b=xserie_b,         # Вторая временная серия
    cross_values=True,         # Анализ пересечений с порогами
    cross_series=True          # Анализ пересечений с сериями
)
df.to_csv('/home/alex/BitcoinScalper/dataframes/RSI_test.csv')
print(df.head())
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"], mode='lines',
    line=dict(color="blue"), name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                            mode='lines', name='RSI',
                            line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["Max"], mode='lines', marker=dict(color='green', size=0.8), name='Max Points'))
fig.add_trace(go.Scatter(
    x=df.index, y=df["Min"], mode='lines', marker=dict(color='red', size=0.8), name='Min Points'))
fig.write_html(r'/home/alex/BitcoinScalper/charts/rsi.html')