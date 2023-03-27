from util import get_data, plot_data
import datetime as dt
import numpy as np
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import sma, ema, macd, bollinger_band_percentage, rate_of_change


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
        df_roc.fillna(method="bfill", inplace=True)
        df_macd = macd(df)
        #print(df_bbp)
        portfolio = df.copy()
        portfolio.iloc[:, :] = np.nan
        current_position = 0
        
        for i in range(1, len(portfolio)):
            if df_ema.iloc[i - 1, 1] < df_ema.iloc[i - 1, 2] and \
                df_ema.iloc[i, 1] >= df_ema.iloc[i, 2]:
                ema_result =  1
            elif df_ema.iloc[i - 1, 1] > df_ema.iloc[i - 1, 2] and \
                df_ema.iloc[i, 1] <= df_ema.iloc[i, 2]:
                ema_result = -1
            else:
                ema_result = 0

            if df_roc.iloc[i, 0] > df_roc.iloc[i - 1 , 0]:
                roc_result = 1
            else:
                roc_result = -1

            if df_macd.iloc[i, 0] > df_macd.iloc[i, 1]:
                macd_result = 1
            else:
                macd_result = -1

            total = ema_result + roc_result + macd_result

            if total >= 2:
                action = 1000 - current_position

            elif total <= -2:
                action = -1000 - current_position

            else:
                action = -current_position

            print(action)
            portfolio.iloc[i, 0] = action
        return portfolio
if __name__ == "__main__":
    ms = ManualStrategy()
    print(ms.testPolicy())
