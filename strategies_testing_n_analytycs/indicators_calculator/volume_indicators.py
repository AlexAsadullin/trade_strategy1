import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly._subplots import make_subplots

class VolumeIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame, fig):
        self.df = df
        self.fig = fig

    def ad(self, length=None, offset=None):
        self.df["ad"] = ta.ad(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            length=length, 
            offset=offset
        )

    def adosc(self, fast=None, slow=None, offset=None):
        self.df["adosc"] = ta.adosc(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            fast=fast, 
            slow=slow, 
            offset=offset
        )

    def cmf(self, length=None, offset=None):
        self.df["cmf"] = ta.cmf(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            length=length, 
            offset=offset
        )

    def efi(self, length=None, offset=None):
        self.df["efi"] = ta.efi(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            length=length, 
            offset=offset
        )

    def mfi(self, length=None, scalar=None, offset=None):
        self.df["mfi"] = ta.mfi(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            length=length, 
            scalar=scalar, 
            offset=offset
        )

    def kvo(self, fast=None, slow=None, signal=None, offset=None):
        self.df = pd.concat([self.df, ta.kvo(
            high=self.df["High"], 
            low=self.df["Low"], 
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            fast=fast, 
            slow=slow, 
            signal=signal, 
            offset=offset)], axis='columns')

    def obv(self, offset=None):
        self.df["obv"] = ta.obv(
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            offset=offset
        )

    def pvi(self, offset=None):
        self.df["pvi"] = ta.pvi(
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            offset=offset
        )

    def pvt(self, offset=None):
        self.df["pvt"] = ta.pvt(
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            offset=offset
        )

    def vp(self, length=None, nbins=None, offset=None):
        self.df = pd.concat([self.df, ta.vp(
            close=self.df["Close"], 
            volume=self.df["Volume"], 
            length=length, 
            nbins=nbins, 
            offset=offset)], axis='columns')

def main(data_read_path: str, data_write_path: str, chart_path: str):
    df = pd.read_csv(data_read_path, index_col=0)
    df = df.reset_index()
    calculator = VolumeIndicatorsCalculator(df=df, fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02))
    
    calculator.ad(length=14, offset=0)
    calculator.adosc(fast=3, slow=10, offset=0)
    calculator.cmf(length=20, offset=0)
    calculator.efi(length=13, offset=0)
    calculator.mfi(length=14, scalar=100, offset=0)
    calculator.kvo(fast=34, slow=55, signal=13, offset=0)
    calculator.obv(offset=0)
    calculator.pvi(offset=0)
    calculator.pvt(offset=0)
    calculator.vp(length=30, nbins=30, offset=0)
    
    calculator.df.to_csv(data_write_path)

if __name__ == '__main__':
    main(data_read_path=r'/home/alex/BitcoinScalper/dataframes/TSLA.csv',
         data_write_path=r'/home/alex/BitcoinScalper/dataframes/TSLA_volume.csv',
         chart_path=r'/home/alex/BitcoinScalper/html_charts/TSLA_volume.html')