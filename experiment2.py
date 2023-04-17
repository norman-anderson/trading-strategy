from util import get_data, plot_data
import datetime as dt
import numpy as np
import pandas as pd
import marketsimcode
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner


def author():
    return "nanderson83"

def experiment2():
    np.random.seed(903863313)
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0

    # impact 0
    sl0 = StrategyLearner(impact=0)
    sl0.add_evidence(symbol="JPM")
    trades_df = sl0.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    sl_portvals_0 = marketsimcode.compute_portvals(trades_df, sv, 0, 0)
    sl_portvals_0 = sl_portvals_0 / sl_portvals_0.iloc[0]

    # impact = 0.005
    sl05 = StrategyLearner(impact=0.005)
    sl05.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df05 = sl05.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    sl_portvals_05 = marketsimcode.compute_portvals(trades_df05, sv, 0, 0.005)
    sl_portvals_05 = sl_portvals_05 / sl_portvals_05.iloc[0]

    # impact = 0.01
    sl1 = StrategyLearner(impact=0.01)
    sl1.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df1 = sl1.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    sl_portvals_1 = marketsimcode.compute_portvals(trades_df1, sv, 0, 0.01)
    sl_portvals_1 = sl_portvals_1 / sl_portvals_1.iloc[0]

    plt.figure(figsize=(18, 9))
    plt.plot(sl_portvals_05, 'Green', label='Impact Value = $0.00')
    plt.plot(sl_portvals_0, 'Red', label='Impact Value = $0.005')
    plt.plot(sl_portvals_1, 'Blue', label='Impact Value = $0.01')
    plt.title('Strategy Learner with Different Impact Values', fontsize=26)
    plt.legend(fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Normalized Price', fontsize=20)
    plt.savefig('experiment2.png')
    plt.clf()

    ### statistics ###
    cr0, mean0, stdev0 = stats(sl_portvals_0)
    cr05, mean05, stdev05 = stats(sl_portvals_05)
    cr1, mean1, stdev1 = stats(sl_portvals_1)

    print("-------Experiment 2---------")
    print("Impact 0.0")
    print("Cumulative return: {:6f}".format(cr0))
    print("Mean of daily returns: {:6f}".format(mean0))
    print("Stdev of daily returns: {:6f}".format(stdev0))
    print("Impact 0.5")
    print("Cumulative return: {:6f}".format(cr05))
    print("Mean of daily returns: {:6f}".format(mean05))
    print("Stdev of daily returns: {:6f}".format(stdev05))
    print("vs.")
    print("Impact 1")
    print("Cumulative return: {:6f}".format(cr1))
    print("Mean of daily returns: {:6f}".format(mean1))
    print("Stdev of daily returns: {:6f}".format(stdev1))
    print("")

def stats(portvals):
    portvals = portvals['Cash']
    daily_returns = (portvals / portvals.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr = (portvals[-1] / portvals[0]) - 1
    mean = daily_returns.mean()
    stdev = daily_returns.std()
    return cr, mean, stdev
