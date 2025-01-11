import os
import plotly.graph_objects as go
import json
import pandas as pd
import numpy as np
#import talib # on windows
import pandas_ta as ta

class Deal:
    def __init__(self, entry_price: float | int, entry_time):
        # entry
        # self.entry_trigger = 
        self.length = 0
        self.entry_time = entry_time
        self.entry_price = entry_price
        # exit
        self.exit_time = 0
        self.exit_price = 0
        # money
        self.profit = 0
    def forced_close(self):
        self.exit_price = -1
        self.exit_time = -1
        self.exit_trigger = -1
        self.length = -1
        # short no comission
        self.profit = -1

class ShortDeal(Deal):
    def close_deal(self, exit_trigger: str, exit_price: float | int, exit_time):
        #self.exit_trigger = 
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_trigger = exit_trigger
        self.length = self.exit_time - self.entry_time
        # short no comission
        self.profit = self.entry_price - self.exit_price
        return self.profit
    def count_profit(self, current_price):
        return self.entry_price - current_price

class LongDeal(Deal):
    def close_deal(self, exit_trigger: str, exit_price: float | int, exit_time):
        #self.exit_trigger = 
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_trigger = exit_trigger
        self.length = self.exit_time - self.entry_time
        # long no comission
        self.profit = self.exit_price - self.entry_price
        return self.profit
    def count_profit(self, current_price):
        return current_price - self.entry_price


class BasicTester:
    def __init__(self, prices_df: pd.DataFrame, max_deal_len: int, min_deal_len: int,
                 long_condition, short_condition,
                 start_deposit: float,
                 long_stop_loss_coef: float, long_take_profit_coef: float,
                 short_stop_loss_coef: float, short_take_profit_coef: float):
        # data
        self.prices_df = prices_df
        # deals
        self.current_deal = 0
        self.deal_len_counter = 0
        self.max_deal_len = max_deal_len
        self.min_deal_len = min_deal_len
        self.now_deal_opened = False
        # conditions
        self.long_condition = long_condition
        self.short_condition = short_condition
        # money
        self.money = start_deposit
        self.deals_history = {'Long': [], 'Short': []}
        '''self.strategy_results = {'keys': []
                                 'LenSoft': [],
                                 'LenHard': [],
                                 'StopLoss': [],
                                 'TakeProfit': [],
                                 }'''
        # exits
        self.long_stop_loss_coef = long_stop_loss_coef # less 1
        self.long_take_profit_coef = long_take_profit_coef # more 1
        self.short_stop_loss_coef = short_stop_loss_coef # more 1
        self.short_take_profit_coef = short_take_profit_coef # less 1
    
    def _check_all_conditions(self, row, i):
        if self.now_deal_opened:
            self._check_if_deal_should_be_closed(row=row, index=i)
        elif self.long_condition:
            self.current_deal = self._open_long_deal(row=row, index=i)
            print('open long index:', i, end=' ')
        elif self.short_condition:
            self.current_deal = self._open_short_deal(row=row, index=i)
            print('open short index:', i, end=' ')

    def _close_any_deal(self, exit_trigger: str, deal, price, i):
        self.deal_len_counter = 0
        self.now_deal_opened = False
        local_profit = deal.close_deal(exit_price=price, exit_trigger=exit_trigger, exit_time=i) 
        self.money += local_profit
        self.current_deal = 0
        return deal

    def _check_if_deal_should_be_closed(self, row: pd.Series, index):
        deal = self.current_deal
        deal_name = str(type(deal)).lower()
        length_condition_soft = self.deal_len_counter > self.max_deal_len
        length_condition_hard = self.deal_len_counter > self.max_deal_len * 1.2
        if 'long' in deal_name:
            takeprofit_condition = row['Close'] >= deal.entry_price * self.long_take_profit_coef
            stoploss_condition = row['Close'] <= deal.entry_price * self.long_stop_loss_coef
        else:
            takeprofit_condition = row['Close'] <= deal.entry_price * self.short_take_profit_coef
            stoploss_condition = row['Close'] >= deal.entry_price * self.long_stop_loss_coef
        if takeprofit_condition:
            deal = self._close_any_deal(deal=deal, exit_trigger='TakeProfit', price=row['Close'], i=index)
            if 'short' in deal_name:
                self.deals_history['Short'].append(deal)
            else:
                self.deals_history['Long'].append(deal)
            
            print('close take profit, profit:', deal.profit, 'len:', self.deal_len_counter)
        elif self.deal_len_counter < self.min_deal_len:
            self.deal_len_counter += 1
        elif stoploss_condition:
            deal = self._close_any_deal(deal=deal, exit_trigger='StopLoss', price=row['Close'], i=index)
            if 'short' in deal_name:
                self.deals_history['Short'].append(deal)
            else:
                self.deals_history['Long'].append(deal)
            print('close stop loss, profit:', deal.profit, 'len:', self.deal_len_counter, 'index:', index)
        elif length_condition_soft and deal.count_profit(current_price=row['Close']) > 0:
            deal = self._close_any_deal(deal=deal, exit_trigger='TimeSoft', price=row['Close'], i=index)
            if 'short' in deal_name:
                self.deals_history['Short'].append(deal)
            else:
                self.deals_history['Long'].append(deal)
            print('close by length, wait till profit:', deal.profit, 'len:', self.deal_len_counter, 'index:', index)
        elif length_condition_hard:
            deal = self._close_any_deal(deal=deal, exit_trigger='TimeHard', price=row['Close'], i=index)
            if 'short' in deal_name:
                self.deals_history['Short'].append(deal)
            else:
                self.deals_history['Long'].append(deal)
            print('close by length, hard cond. profit:', deal.profit, 'len:', self.deal_len_counter, 'index:', index)
        else:
            self.deal_len_counter += 1

    def _open_long_deal(self, row: pd.Series, index: int):
        deal = LongDeal(entry_price=row['Close'], entry_time=index)
        self.deal_len_counter = 0
        self.now_deal_opened = True
        return deal
    
    def _open_short_deal(self, row: pd.Series, index: int):
        deal = ShortDeal(entry_price=row['Close'], entry_time=index)
        self.deal_len_counter = 0
        self.now_deal_opened = True
        return deal
    
    def pos_neg_analysys(self, currency: str = 'rub'):
        pos_deals, neg_deals, zeros, profit_pos, profit_neg, exit_take, exit_stop, exit_timesoft, exit_timehard = 0, 0, 0, 0, 0, 0, 0, 0, 0
        total = {'Long': {},
                 'Short': {}}
 
        for key in ['Long', 'Short']:
            for deal in self.deals_history[key]:
                if deal.profit > 0:
                    profit_pos += deal.profit
                    pos_deals += 1
                elif deal.profit < 0:
                    profit_neg += deal.profit
                    neg_deals += 1
                else:
                    zeros += 1
                if deal.exit_trigger == 'TakeProfit':
                    exit_take += 1
                elif deal.exit_trigger == 'TimeSoft':
                    exit_stop += 1
                elif deal.exit_trigger == 'StopLoss':
                    exit_timesoft += 1
                elif deal.exit_trigger == 'TimeHard':
                    exit_timehard += 1

            total[key]['positive'] = pos_deals
            total[key]['negative'] = neg_deals
            total[key]['zeros'] = zeros
            total[key]['abs_profit'] = profit_neg + profit_pos
            total[key]['positive_profit'] = profit_pos
            total[key]['negative_profit'] = profit_neg
            total[key]['exits by takeprofit'] = exit_take
            total[key]['exits by stop loss'] = exit_stop
            total[key]['exits by time toft'] = exit_timesoft
            total[key]['exits by time hard'] = exit_timehard
        print(f'absolute profit: {total['Long']['abs_profit'] + total['Short']['abs_profit']}')
        return total

    def derivative(self, target_col_name: str = 'Close'):
        der =  np.gradient(self.prices_df[target_col_name])
        derivative = pd.DataFrame({'Derivative': der}, index=self.prices_df.index)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.prices_df.index, y=self.prices_df['Close'],
                                mode='lines', name='Close',
                                line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.prices_df.index, y=derivative['Derivative'],
                                mode='lines', name='Derivative',
                                line=dict(color='red')))
        fig.write_html('derivative.html')
        return derivative

class BybitGlassMonitor:
    def __init__(self):
        self.glasses_list = []
    def read_json_history(self):
        for filename in os.listdir('bitcoint_history_glass'):
            with open(f'bitcoint_history_glass\\{filename}', 'r') as f:
                data = json.load(f)
                for glass in data:
                    result = {'bid': [], 'ask': []}
                    result['symbol'] = glass['result']['s']
                    result['timestamp'] = glass['result']['ts']
                    for i in glass['result']['b']:
                        result['bid'].append([float(i[0]), float(i[-1])])
                    for i in glass['result']['a']:
                        result['ask'].append([float(i[0]), float(i[-1])])
                self.glasses_list.append(result)

    def glass_unit(self, glass: dict):
        ask_total_amount = 0
        bid_total_amount = 0
        report = {}
        for ask in glass['ask']:
            ask_total_amount += ask[0] * ask[-1]
        for bid in glass['bid']:
            bid_total_amount += bid[0] * bid[-1]
        report['total_ask'] = ask_total_amount
        report['total_bid'] = bid_total_amount 
        report['ask_bid_diff'] = ask_total_amount - bid_total_amount
        return report

"""
import plotly.graph_objects as go
glass_tester = BybitGlassMonitor()
glass_tester.read_json_history()
visual = {'total_ask': [], 'total_bid': [], 'ask_bid_diff': []}
for i in glass_tester.glasses_list:
    report = glass_tester.glass_unit(glass=i)
    visual['ask_bid_diff'].append(report['ask_bid_diff'])
    visual['total_ask'].append(report['total_ask'])
    visual['total_bid'].append(report['total_bid'])

df = pd.DataFrame.from_dict(visual)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['total_ask'],
                            mode='lines', name='ask money',
                            line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=df['total_bid'],
                            mode='lines', name='bid_money',
                            marker=dict(color='red', size=8)))
fig.write_html('glass.html')"""