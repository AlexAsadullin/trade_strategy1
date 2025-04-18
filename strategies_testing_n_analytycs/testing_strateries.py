from strategy_tester_parent import *
import pandas_ta as ta

class RsiTester(BasicTester):
    def delta_rsi_test(self, rsi_delta_long, rsi_delta_short, delta_len: int):
        self.now_deal_opened = False
        self.deals_history = {'Long': [], 'Short': []}
        self.deal_len_counter = 0
        rsi_queue, volume_queue, close_queue = [], [], []

        for i, row in self.prices_df.iloc[:delta_len].iterrows():
            rsi_queue.append(row['RSI'])
            volume_queue.append(row['Volume'])
            close_queue.append(row['Close'])
        for i, row in self.prices_df.iloc[delta_len:].iterrows():
            rsi_delta = rsi_queue[-1] - rsi_queue[0]
            volume_delta = volume_queue[-1] - volume_queue[0]
            close_delta = close_queue[-1] - volume_queue[0]
            # conditions 
            self.long_condition = rsi_delta >= rsi_delta_long
            self.short_condition = rsi_delta <= rsi_delta_short

            self._check_all_conditions(row=row, i=i)

            rsi_queue = rsi_queue[1:]
            rsi_queue.append(row['RSI'])
            volume_queue = volume_queue[1:]
            volume_queue.append(row['Volume'])
            close_queue = close_queue[1:]
            close_queue.append(row['Close'])
        self.now_deal_opened = False

    def rsi_simple_test(self, rsi_long, rsi_short, rsi_len: int): 
        self.now_deal_opened = False
        self.deals_history = {'Long': [], 'Short': []}
        self.deal_len_counter = 0
        self.prices_df['RSI'] = ta.RSI(real=self.prices_df['Close'], timeperiod=rsi_len)
        for i, row in self.prices_df.iterrows():
            self.long_condition = row['RSI'] < rsi_long 
            self.short_condition = row['RSI'] > rsi_short
            
            self._check_all_conditions(row=row, i=i)
        self.now_deal_opened = False
        self.current_deal.forced_close()
        
        # TODO: покупаем не 1 шт акций а больше - в зависимости от "уверенности"
if __name__ == '__main__':
    df = pd.read_csv(r'/home/alex/BitcoinScalper/dataframes/TSLA_RSI.csv', index_col=0)
    df = df.dropna()
    print(df.head())

    tester = RsiTester(prices_df=df, min_deal_len=0, max_deal_len=30, start_deposit=1000,
                    long_condition=0, short_condition=0,
                    long_stop_loss_coef=0.7, long_take_profit_coef=1.3,
                    short_stop_loss_coef=1.3, short_take_profit_coef=0.7)

    df = tester.prices_df
    tester.delta_rsi_test(rsi_delta_long=30, rsi_delta_short=45, delta_len=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                    mode='lines', name='Close',
                                    line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                    mode='lines', name='Close',
                                    line=dict(color='blue')))
    
    print(tester.pos_neg_analysys())
    #fig.show()
    #tester.delta_hilbert_test(hilbert_delta_long=-66, hilbert_delta_short=66, delta_len=12)
