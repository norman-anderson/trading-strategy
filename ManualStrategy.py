from util import get_data, plot_data
import datetime as dt
import numpy as np
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import sma, rate_of_change, bollinger_band_percentage


def author():
    return "nanderson83"


class ManualStrategy:
    def testPolicy(self, symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
        df = get_data([symbol], pd.date_range(sd, ed))
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        prices = df[symbol]
        prices_normed = prices / prices.iloc[0]

        # indicators
        s, s_ratio = sma(prices_normed)
        upper_band, lower_band, bbp = bollinger_band_percentage(prices_normed)
        roc = rate_of_change(prices_normed)

        portfolio = pd.DataFrame(index=prices_normed.index, columns=['JPM'])
        portfolio.iloc[:, :] = np.nan
        current_position = 0
        for i in range(portfolio.shape[0] - 1):
            if current_position <= 0 and (s_ratio.iloc[i] < 0.6 or bbp.iloc[i] < 0.2 or roc.iloc[i] < -0.2):
                action = 1000 - current_position
                portfolio.iloc[i, 0] = action
                current_position += action

            elif current_position >= 0 and (s_ratio.iloc[i] > 1.4 or bbp.iloc[i] > 0.8 or roc.iloc[i] > 0.2):
                action = -1000 - current_position
                portfolio.iloc[i, 0] = action
                current_position += action

        portfolio.dropna(inplace=True)
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

    print("Manual Strategy")
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
    is_trades = ms.testPolicy()

    is_manual_portvals = compute_portvals(is_trades)
    is_benchmark = ms.benchmark()
    stats(is_manual_portvals, is_benchmark)
    chart(is_trades, is_manual_portvals, is_benchmark)

    # Out of Sample
    os_trades = ms.testPolicy(sd=dt.datetime(2010, 1, 1),
                           ed=dt.datetime(2011, 12, 31))
    os_manual_portvals = compute_portvals(os_trades)
    os_benchmark=ms.benchmark(sd=dt.datetime(2010, 1, 1),
                           ed=dt.datetime(2011, 12, 31))
    stats(os_manual_portvals, os_benchmark)
    chart(os_trades, os_manual_portvals, os_benchmark)


if __name__ == "__main__":
    report()


