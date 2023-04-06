from util import get_data, plot_data
import datetime as dt
import numpy as np
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import ema, macd, rate_of_change


def author():
    return "nanderson83"


class ManualStrategy:
    def testPolicy(self, symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
        df = get_data([symbol], pd.date_range(sd, ed))
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.drop(['SPY'], axis=1, inplace=True)

        # indicators
        df_ema = ema(df)
        df_roc = rate_of_change(df)
        df_macd = macd(df)

        portfolio = df.copy()
        portfolio.iloc[:, :] = np.nan
        current_position = 0
        for date in portfolio.index:
            #print(date)
            vote = df_macd.loc[date, 'signal'] + df_ema.loc[date, 'signal'] + df_roc.loc[date, 'signal']
            if vote >= 2:
                action = 1000 - current_position
            elif vote <= -2:
                action = -1000 - current_position
            else:
                action = -current_position
            portfolio.loc[date, 'JPM'] = action
            print(action)
        return portfolio
if __name__ == "__main__":
    ms = ManualStrategy()
    ms.testPolicy()
