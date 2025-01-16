import pandas as pd
import pandas_ta as ta

class CyclesIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def ebsw(self, length, fast, slow, signal, scalar, offset):
        self.df["ebsw"] = ta.ebsw(
            close=self.df["Close"],  
            length=length,         
            fast=fast,             
            slow=slow,            
            signal=signal,           
            scalar=scalar,         
            offset=offset            
        )

def main(data_read_path, data_write_path, chart_path):
    df = pd.read_csv(data_read_path).drop(['Date'], axis=1)
    calculator = CyclesIndicatorsCalculator(df)
    calculator.ebsw(length=20, fast=4, slow=10, signal=6, scalar=1.0, offset=0)
    df.to_csv(data_write_path)

if __name__ == '__main__':
    main(data_read_path=r"/home/alex/BitcoinScalper/dataframes/TSLA.csv",
        data_write_path=r'/home/alex/BitcoinScalper/dataframes/TSLA_cycles.csv',
        chart_path=r"/home/alex/BitcoinScalper/html_charts/TSLA_performance.html")