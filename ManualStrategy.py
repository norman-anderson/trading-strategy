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
            current_position += action
            portfolio.loc[date, 'JPM'] = action
        return portfolio

    def benchmark(self, symbol="JPM",
                        sd=dt.datetime(2008, 1, 1),
                        ed=dt.datetime(2009, 12, 31), sv=100000):
        df_trades = self.testPolicy('JPM', sd, ed, sv)
        ms_portval = compute_portvals(df_trades, start_val=sv, impact=0, commission=0)
        ms_normed = ms_portval / ms_portval['Cash'][0]

        df_benchmark = df_trades.copy()
        df_benchmark.iloc[:] = 0
        df_benchmark.iloc[0, 0] = 1000

        benchmark_portval = compute_portvals(orders=df_benchmark, start_val=sv, commission=9.95, impact=0.005)
        #benchmark_normed = benchmark_portval / benchmark_portval['Cash'][0]

        return benchmark_portval


def stats(manual, benchmark):
    manual = manual['Cash']
    benchmark = benchmark['Cash']
    manual_cr = manual[-1] / manual[0] - 1
    manual_daily_return = (manual / manual.shift(1) - 1).iloc[1:]
    manual_stdev = manual_daily_return.std()
    manual_mean = manual_daily_return.mean()

    # Benchmark Stats
    benchmark_cr = benchmark[-1] / benchmark[0] - 1
    benchmark_daily_return = (benchmark / benchmark.shift(1) - 1).iloc[1:]
    benchmark_mean = benchmark_daily_return.mean()
    benchmark_stdev = benchmark_daily_return.std()

    print("Theoretically Optimal Strategy")
    print("Cumulative return: {:6f}".format(manual_cr))
    print("Mean of daily returns: {:6f}".format(manual_mean))
    print("Stdev of daily returns: {:6f}".format(manual_stdev))
    print("vs.")
    print("Benchmark")
    print("Cumulative return: {:6f}".format(benchmark_cr))
    print("Mean of daily returns: {:6f}".format(benchmark_mean))
    print("Stdev of daily returns: {:6f}".format(benchmark_stdev))

def chart(trades, manual, benchmark):
    long = []
    short = []
    current = 0
    last_action = 'OUT'
    for date in trades.index:
        current += trades.loc[date, 'JPM']
        if current < 0:
            if last_action == 'OUT' or last_action == 'LONG':
                last_action = 'SHORT'
                short.append(date)
        elif current > 0:
            if last_action == 'OUT' or last_action == 'SHORT':
                last_action = 'LONG'
                long.append(date)
        else:
            last_action = 'OUT'

        # normalize
    benchmark['Cash'] = benchmark['Cash'] / benchmark['Cash'][0]
    manual['Cash'] = manual['Cash'] / manual['Cash'][0]

    plt.figure(figsize=(14,8))
    plt.title("ManualStragety")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=30)
    plt.grid()
    plt.plot(benchmark, label="benchmark", color = "green")
    plt.plot(manual, label="manual", color = "red")

    for date in short:
            plt.axvline(date, color = "black")
    for date in long:
            plt.axvline(date, color = "blue")

    plt.legend()
    #plt.savefig("manual.png")
    plt.show()
    plt.clf()

def report():
    # In Sample
    ms = ManualStrategy()
    trades = ms.testPolicy()

    manual_portvals = compute_portvals(trades)
    benchmark = ms.benchmark()
    stats(manual_portvals, benchmark)
    chart(trades, manual_portvals, benchmark)



    # Out of Sample


if __name__ == "__main__":
    report()

