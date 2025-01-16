import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly._subplots import make_subplots


class MomentumIndicatorsCalculator:
    def __init__(self, df:pd.DataFrame, fig):
        self.df = df
        self.fig = fig

    def ao(self, fast: int = 5, slow: int = 34, offset: int = 0):
        self.df["ao"] = ta.ao(
            high=self.df["High"], 
            low=self.df["Low"], 
            fast=fast, 
            slow=slow, 
            offset=offset
        )

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, talib: bool = False, offset: int = 0):
        macd = ta.macd(
            close=self.df["Close"], 
            fast=fast, 
            slow=slow, 
            signal=signal, 
            talib=talib, 
            offset=offset
        )
        self.df["macd"] = macd["MACD_12_26_9"]
        self.df["macd_signal"] = macd["MACDs_12_26_9"]
        self.df["macd_hist"] = macd["MACDh_12_26_9"]

    def stoch(self, k: int = 14, d: int = 3, smooth_k: int = 3, drift: int = 1, offset: int = 0):
        stoch = ta.stoch(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            k=k, 
            d=d, 
            smooth_k=smooth_k, 
            drift=drift, 
            offset=offset
        )
        self.df["stoch_k"] = stoch["STOCHk_14_3_3"]
        self.df["stoch_d"] = stoch["STOCHd_14_3_3"]

    def cci(self, length: int = 20, c: float = 0.015, talib: bool = False, drift: int = 1, offset: int = 0):
        self.df["cci"] = ta.cci(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            length=length, 
            c=c, 
            talib=talib, 
            drift=drift, 
            offset=offset
        )

    def willr(self, length: int = 14, talib: bool = False, drift: int = 1, offset: int = 0):
        self.df["willr"] = ta.willr(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            length=length, 
            talib=talib, 
            drift=drift, 
            offset=offset
        )

    def stc(self, fast: int = 23, slow: int = 50, factor: float = 0.5, drift: int = 1, offset: int = 0):
        stc_df = ta.stc(
            close=self.df["Close"], 
            fast=fast, 
            slow=slow, 
            factor=factor, 
            drift=drift, 
            offset=offset
        )
        self.df = pd.concat([self.df, stc_df], axis='columns')

    def tsi(self, fast: int = 25, slow: int = 13, scalar: int = 100, drift: int = 1, offset: int = 0):
        tsi_df = ta.tsi(
            close=self.df["Close"], 
            fast=fast, 
            slow=slow, 
            scalar=scalar, 
            drift=drift, 
            offset=offset
        )
        self.df = pd.concat([self.df, tsi_df], axis='columns')

    def fisher(self, length: int = 9, signal: int = 1, offset: int = 0):
        fisher = ta.fisher(
            high=self.df["High"], 
            low=self.df["Low"], 
            length=length, 
            signal=signal, 
            offset=offset
        )
        self.df = pd.concat([self.df, fisher], axis='columns')

    def kst(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                window1: int = 10, window2: int = 10, window3: int = 10, window4: int = 15,
                signal: int = 9, offset: int = 0):
        kst = ta.kst(
            close=self.df["Close"], 
            roc1=roc1, 
            roc2=roc2, 
            roc3=roc3, 
            roc4=roc4, 
            window1=window1, 
            window2=window2, 
            window3=window3, 
            window4=window4, 
            signal=signal, 
            offset=offset
        )
        self.df = pd.concat([self.df, kst], axis='columns')

    def rsi(self, length: int, drift: int, xa: int, xb: int, offset: int=0,
            scalar:int=100, cross_values:bool=True, cross_series:bool=True):
        xserie = self.df["Open"]  # Дополнительная серия
        xserie_a = self.df["High"]  # Первая дополнительная серия
        xserie_b = self.df["Low"]   # Вторая дополнительная серия

        self.df['rsi'] = ta.rsi(
            close=self.df["Close"],         # Основная временная серия
            length=length,                 # Длина окна RSI
            scalar=scalar,                # Масштабирование (обычно 100 для RSI)
            drift=drift,                   # Шаг для diff
            offset=offset,                  # Без смещения
            #signal_indicators=True,    # Включение сигналов
            xa=xa,                     # Верхний порог RSI
            xb=xb,                     # Нижний порог RSI
            xserie=xserie,             # Дополнительная временная серия
            xserie_a=xserie_a,         # Первая временная серия
            xserie_b=xserie_b,         # Вторая временная серия
            cross_values=cross_values,         # Анализ пересечений с порогами
            cross_series=cross_series          # Анализ пересечений с сериями
        )



def main(data_read_path:str, data_write_path:str, chart_path:str):
    df = pd.read_csv(data_read_path, index_col=0)
    df = df.reset_index()
    calculator = MomentumIndicatorsCalculator(df=df, fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02))
    calculator.ao(fast=5, slow=34, offset=0)
    calculator.rsi(length=18, scalar=100, drift=1, offset=0, xa=80, xb=20)
    calculator.macd(fast=12, slow=26, signal=9, talib=False, offset=0)
    calculator.stoch(k=14, d=3, smooth_k=3, drift=1, offset=0)
    calculator.cci(length=20, c=0.015, talib=False, drift=1, offset=0)
    calculator.willr(length=14, talib=False, drift=1, offset=0)
    calculator.stc(fast=23, slow=50, factor=0.5, drift=1, offset=0)
    calculator.tsi(fast=25, slow=13, scalar=100, drift=1, offset=0)
    calculator.fisher(length=9, signal=1, offset=0)
    calculator.kst(
        roc1=10, roc2=15, roc3=20, roc4=30, 
        window1=10, window2=10, window3=10, window4=15, 
        signal=9, offset=0
    )

    calculator.df.to_csv(data_write_path)

    f"""ig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
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
    fig.write_html(chart_path)"""

if __name__ == '__main__':
    main(data_read_path=r'/home/alex/BitcoinScalper/dataframes/TSLA.csv',
         data_write_path=r'/home/alex/BitcoinScalper/dataframes/TSLA_momentum.csv',
         chart_path=r'/home/alex/BitcoinScalper/html_charts/rsi.html')