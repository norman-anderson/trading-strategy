from util import get_data, plot_data
import datetime as dt
import numpy as np
import marketsimcode
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner


def author():
    return "nanderson83"


def experiment1(sd, ed):
    np.random.seed(903863313)
    symbol = 'JPM'
    sv = 100000
    impact = 0.005
    commission = 9.95


    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol, sd, ed, sv)
    ms_portvals = marketsimcode.compute_portvals(df_trades, sv, 9.95, 0.005)
    ms_portvals = ms_portvals / ms_portvals.iloc[0]

    bench_portvals = ms.benchmark(symbol, sd, ed, sv)
    bench_portvals = bench_portvals / bench_portvals.iloc[0]

    sl = StrategyLearner(impact=0.005)
    sl.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    sl_df = sl.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    sl_portvals = marketsimcode.compute_portvals(sl_df, sv, 9.95, 0.005)
    sl_portvals = sl_portvals / sl_portvals.iloc[0]

    plt.figure(figsize=(18, 9))
    plt.plot(ms_portvals, 'Blue', label='Manual Strategy')
    plt.plot(bench_portvals, 'Green', label='Benchmark')
    plt.plot(sl_portvals, 'Red', label='Strategy Learner')
    plt.title('Normalized Value of Strategy Learner vs Manual Strategy vs Benchmark', fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Normalized Price', fontsize=20)
    plt.legend()
    plt.savefig('experiment1.png')
    plt.clf()

    ### statistics ###
    ms_cr, ms_mean, ms_stdev = stats(ms_portvals)
    bench_cr, bench_mean, bench_stdev = stats(bench_portvals)
    sl_cr, sl_mean, sl_stdev = stats(sl_portvals)

    print("-------Experiment 1---------")
    print("Strategy Learner")
    print("Cumulative return: {:6f}".format(sl_cr))
    print("Mean of daily returns: {:6f}".format(sl_mean))
    print("Stdev of daily returns: {:6f}".format(sl_stdev))
    print("Manual Strategy")
    print("Cumulative return: {:6f}".format(ms_cr))
    print("Mean of daily returns: {:6f}".format(ms_mean))
    print("Stdev of daily returns: {:6f}".format(ms_stdev))
    print("vs.")
    print("Benchmark")
    print("Cumulative return: {:6f}".format(bench_cr))
    print("Mean of daily returns: {:6f}".format(bench_mean))
    print("Stdev of daily returns: {:6f}".format(bench_stdev))
    print("")


def stats(portvals):
    portvals = portvals['Cash']
    daily_returns = (portvals / portvals.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr = (portvals[-1] / portvals[0]) - 1
    mean = daily_returns.mean()
    stdev = daily_returns.std()
    return cr, mean, stdev
