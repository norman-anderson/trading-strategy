""""""
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Norman Anderson		  	   		  		 			  		 			     			  	 
GT User ID: nanderson83  		  	   		  		 			  		 			     			  	 
GT ID: 903863313		  	   		  		 			  		 			     			  	 
"""

import datetime as dt
import random

import pandas as pd
import util as ut

from RTLearner import RTLearner
from BagLearner import BagLearner
from indicators import sma, ema, macd, bollinger_band_percentage, rate_of_change


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        self.learner = BagLearner(learner=RTLearner, kwargs={'leaf_size': 5},
                                  bags=20)
        random.seed(903863313)
        self.lookback = 14

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        if self.verbose:
            print(prices)

        # example use with new colname
        volume_all = ut.get_data(
            syms, dates, colname="Volume"
        )  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)

        s, m, r = sma(prices), macd(prices, symbol), rate_of_change(prices, symbol)
        ind = pd.concat((s, m, r), axis=1)
        x_train = ind.values
        y_train = (prices.shift(-self.lookback) / prices) - 1
        y_train[y_train > 0] = prices.shift(-self.lookback) / (prices * (1.0 + 2 * self.impact)) - 1.0
        y_train[y_train < 0] = prices.shift(-self.lookback) / (prices * (1.0 - 2 * self.impact)) - 1.0
        y_train = y_train.values

        buy, sell = 0.05, -0.05
        for i in range(len(y_train)):
            if y_train[i] > 1.08 + self.impact:
                y_train[i] = 1
            elif y_train[i] < 0.92 - self.impact:
                y_train[i] = -1
            else:
                y_train[i] = 0

        self.learner.add_evidence(x_train, y_train)


    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        # your code should return the same sort of data
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        s, m, r = sma(prices), macd(prices, symbol), rate_of_change(prices, symbol)
        ind = pd.concat((s, m, r), axis=1)
        x_test = ind.values
        y_test = self.learner.query(x_test)
        trades = pd.DataFrame(0, columns=prices.columns, index=prices.index)
        shares = 0
        for i in range(len(trades)):
            if y_test[i] == 1:
                trades[symbol].iloc[i] = 1000 - shares
                shares = 1000
            elif y_test[i] == -1:
                trades[symbol].iloc[i] = - shares - 1000
                shares = -1000

        return trades

if __name__ == "__main__":
    #print("One does not simply think up a strategy")
    sl = StrategyLearner()
    sl.add_evidence()
