import pandas as pd
import pandas_ta as ta

class OverlapIndicatorsCalculator:
    def __init__(self, df):
        self.df = df

    def add_ema(self, length=20, talib=False, offset=0):
        self.df["ema"] = ta.ema(
            close=self.df["Close"],
            length=length,
            talib=talib,
            offset=offset
        )

    def add_supertrend(self, length=10, multiplier=3.0, offset=0):
        st = ta.supertrend(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            length=length,
            multiplier=multiplier,
            offset=offset
        )
        self.df["supertrend"] = st["SUPERTd_10_3.0"]

    def add_vwap(self, offset=0):
        self.df["vwap"] = ta.vwap(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            volume=self.df["Volume"],
            offset=offset
        )

    def add_ichimoku(self, tenkan=9, kijun=26, senkou=52, lookahead=False, offset=0):
        ichimoku_result = ta.ichimoku(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            tenkan=tenkan,
            kijun=kijun,
            senkou=senkou,
            lookahead=lookahead,
            offset=offset
        )
        self.df["tenkan_sen"] = ichimoku_result[0]["ITS_9"]
        self.df["kijun_sen"] = ichimoku_result[0]["IKS_26"]
        self.df["senkou_span_a"] = ichimoku_result[0]["ISA_9"]
        self.df["senkou_span_b"] = ichimoku_result[0]["ISB_26"]

    def add_kama(self, length=10, fast=2, slow=30, offset=0):
        self.df["kama"] = ta.kama(
            close=self.df["Close"],
            length=length,
            fast=fast,
            slow=slow,
            offset=offset
        )

    def add_hma(self, length=16, offset=0):
        self.df["hma"] = ta.hma(
            close=self.df["Close"],
            length=length,
            offset=offset
        )

    def add_t3(self, length=5, volume_factor=0.7, talib=False, offset=0):
        self.df["t3"] = ta.t3(
            close=self.df["Close"],
            length=length,
            volume_factor=volume_factor,
            talib=talib,
            offset=offset
        )

    """def add_mcgd(self, length=10, offset=0):
        self.df["mcgd"] = ta.mcgd(
            close=self.df["Close"],
            length=length,
            offset=offset
        )"""

    def add_alma(self, length=9, sigma=6, offset=0, drift=1):
        self.df["alma"] = ta.alma(
            close=self.df["Close"],
            length=length,
            sigma=sigma,
            offset=offset,
            drift=drift
        )

    def add_wma(self, length=9, offset=0):
        self.df["wma"] = ta.wma(
            close=self.df["Close"],
            length=length,
            offset=offset
        )


def main(data_read_path, data_write_path, chart_path):
    #df.reset_index(inplace=True)
    df = pd.read_csv(data_read_path).drop(['Date'], axis=1)
    df.index = pd.to_datetime(df.index)
    month_index = df.index.to_period('M')

    calculator = OverlapIndicatorsCalculator(df)

    calculator.add_ema(length=20, talib=False, offset=0)
    calculator.add_supertrend(length=10, multiplier=3.0, offset=0)
    calculator.add_vwap(offset=0)
    calculator.add_ichimoku(tenkan=9, kijun=26, senkou=52, lookahead=False, offset=0)
    calculator.add_kama(length=10, fast=2, slow=30, offset=0)
    calculator.add_hma(length=16, offset=0)
    calculator.add_t3(length=5, volume_factor=0.7, talib=False, offset=0)
    #calculator.add_mcgd(length=10, offset=0)
    calculator.add_alma(length=9, sigma=6, offset=0, drift=1)
    calculator.add_wma(length=9, offset=0)

    calculator.df.to_csv(data_write_path)

if __name__ == "__main__":
    data_read_path=r"/home/alex/BitcoinScalper/dataframes/TSLA.csv",
    data_write_path=r"/home/alex/BitcoinScalper/dataframes/TSLA_overlap.csv",
    chart_path=r"/home/alex/BitcoinScalper/html_charts/TSLA_performance.html"