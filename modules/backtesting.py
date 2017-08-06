import pandas as pd
import numpy as np
import sqlite3 as lite
import matplotlib.pyplot as plt
import quandl
from collections import OrderedDict
import time
class Report:
    def __init__(self, pinfo, data):
        self.data = data
        self.info = pinfo
        self._receipt = []
        
    def buy(self, position, date, price):
        if position.lower() == 'long':  pos = 1 
        elif position.lower() == 'short': pos = -1
        self.trade = [pos, date, price]

    def sell(self, date, price):
        self.trade += [date,price]
        self._receipt.append(self.trade)
    
    def result(self):
        return BackTesting.get_result(self.receipt, self.info['name'])

    def plot(self):
        receipt = self.receipt
        fig, (ax) = plt.subplots(2,1, figsize=(12,10))
        ax[0].plot(self.data.Settle)
        for _, entrydate, exitdate in receipt[['entrydate','exitdate']].itertuples():
            ax[0].axvspan(entrydate, exitdate, facecolor='y', alpha=0.3)

        ax[1].plot(receipt.entrydate, receipt.cumprofit)
        ax[1].plot(receipt.entrydate, receipt.cumprofit.cummax())

        return ax[0]
    
    @property
    def receipt(self):
        receipt = pd.DataFrame(
            data = self._receipt,
            columns=['position', 'entrydate', 'entry', 'exitdate', 'exit'])
        receipt['profit'] = receipt.position*(receipt.exit - receipt.entry)/self.info['tick_unit']
        receipt['cumprofit'] = receipt.profit.cumsum()
        receipt['drawdown'] = receipt.cumprofit.cummax() - receipt.cumprofit
        #receipt['drawdown'] = (1- receipt.cumprofit / receipt.cumprofit.cummax())*100 
        return receipt


class BackTesting:
    API_KEY = 'UzB-e5CDdoACq4ENxbVS'
    META_PATH = 'data/SCF-meta.csv'
    INFO_PATH = 'data/db.sqlite3'

    def __init__(self, strategy=None, principal=10000):
        quandl.ApiConfig.api_key = BackTesting.API_KEY
        self.meta = self._get_meta_info(BackTesting.META_PATH, BackTesting.INFO_PATH)
        
        self.principal = principal
        self.reports = []

        if strategy:
            self.strategy = strategy           
        else: 
            self.strategy = self.default_strategy

        
    def run(self):
        print("trading started. it takes few minutes...")
        counter = 1
        for symbol, row in self.meta.iterrows():
            data = quandl.get(row.iid)
            report = Report(row, data)
            self.strategy(report)
            self.reports.append(report)

            if counter % 10 == 1:
                print(f'processing..({counter}/{len(self.meta)})')
            counter += 1
        print("Done")

    def _get_meta_info(self, metapath, infopath):
        """
        quandl 사이트의 SCF에서 제공하는 해외선물 코드 정보와 eBest api에서 제공하는
        상품 정보를 종합하여 백테스팅에 사용될 종목 정보 데이터 프레임을 구성함
        """
        con = lite.connect(infopath)
        info = pd.read_sql('select * from trading_product', con).set_index('group', drop=True)
        meta = pd.read_csv(metapath)
        meta['iid'] = 'SCF/' + meta['Exchange'] + '_' + meta['Symbol'] + '1_OB'
        meta = meta[~meta['Ebest Symbol'].str.contains('None')]\
                .set_index('Ebest Symbol')[['iid','name']]

        meta[['tick_unit', 'tick_value', 'margin', 'commission']] = \
            info[['tick_unit', 'tick_value', 'keep_margin', 'commission']]

        return meta
    
    def result(self):
        """
        결과 분석 리포트
        """
        total = pd.concat([item.receipt for item in self.reports])[['entrydate','exitdate','profit']]
        total.sort_values('entrydate', inplace=True)
        total.reset_index(drop=True, inplace=True)
        total['cumprofit'] = total.profit.cumsum()
        total['drawdown'] = total.cumprofit.cummax() - total.cumprofit

        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        ax.plot(total.entrydate, total.cumprofit)
        ax.plot(total.entrydate, total.cumprofit.cummax())
        plt.show()    
        return BackTesting.get_result(total, 'Total')

    def total_receipt(self):
        return pd.concat([item.result() for item in self.reports])

    @staticmethod
    def get_result(receipt, name):
        #receipt = self.receipt
        mdd = 100*receipt.drawdown.max()/receipt.cumprofit.max()
        cum_profit = receipt.cumprofit.iloc[-1]
        ave_profit = receipt.profit.mean()
        rng = receipt.exitdate.max() - receipt.entrydate.min()
        rng = rng.days/365
        cagr = pow(cum_profit, 1/rng) - 1 if cum_profit > 0 else 0
        win_rate = 100 * receipt.profit[receipt.profit >= 0].count() \
                    / receipt.profit.count()
        max_lose = receipt.profit.min()
        max_profit = receipt.profit.max()
        ave_win = receipt.profit[receipt.profit >= 0].mean()
        ave_lose = receipt.profit[receipt.profit < 0].mean()
        pl_ratio = abs(ave_profit/ave_lose)
        num_trade = len(receipt) / rng # 연평균 매매 횟수
        report = OrderedDict(
            cum_profit=cum_profit, ave_profit=ave_profit, cagr=cagr, mdd=mdd,
            win_rate=win_rate, max_lose=max_lose, max_profit=max_profit, 
            ave_win=ave_win, ave_lose=ave_lose, pl_ratio=pl_ratio, num_trade=num_trade
        )
        return pd.DataFrame(report, index=[name])
    
    @staticmethod
    def default_strategy(report):
        """
        default: buy only ma cross system
        진입: 20종가이평이 60종가이평 돌파상승 한 다음날 시가 진입
        청산: 20종가이평이 60종가이평 돌파하락 한 다음날 시가 청산
        """
        data = report.data
        data['ma20'] = data.Settle.rolling(20).mean()
        data['ma60'] = data.Settle.rolling(60).mean()
        data['cross'] = (data.ma20 > data.ma60).astype('int')
        data['signal'] = data.cross.diff().shift(1)
        data.dropna(inplace=True)
        signals = data[(data.signal == 1) | (data.signal == -1)]
        
        for date, price, signal in signals[['Open','signal']].itertuples():
            if signal == 1:
                report.buy('Long', date, price)
            elif signal == -1:
                report.sell(date, price)