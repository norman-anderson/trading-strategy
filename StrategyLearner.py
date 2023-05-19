import datetime as dt
import random
import pandas as pd
from util import get_data
from QLearner import QLearner
import indicators


def author():
    return "nanderson83"


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
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        self.learner = QLearner(num_states=1000,
                                num_actions=3,
                                alpha=0.2,
                                gamma=0.9,
                                rar=0.9,
                                radr=0.99,
                                dyna=0,
                                verbose=False)
        random.seed(5000000)

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

        ema_20, ema_50, macd, roc = get_discretized(sd, ed, symbol)

        prices = get_prices(sd, ed, symbol)
        portfolio = pd.DataFrame(index=prices.index, columns=[symbol])
        portfolio[:] = 0
        date = prices.index

        current_position = prev_position = 0
        current_cash = prev_cash = sv

        for i in range(portfolio.shape[0] - 1):
            s_prime = get_s_prime(current_position, ema_20.loc[date[i]],
                                  ema_50.loc[date[i]], macd.loc[date[i]],
                                  roc.loc[date[i]])

            reward = current_position * prices.loc[date[i], symbol] + current_cash - prev_position * prices.loc[date[i], symbol] - prev_cash

            vote = self.learner.query(s_prime, reward)
            if vote == 0:
                action = -1000 - current_position
            elif vote == 1:
                action = -current_position
            else:
                action = 1000 - current_position

            prev_position = current_position
            current_position += action
            portfolio.iloc[i, 0] = action

            if action > 0:
                impact = self.impact
            else:
                impact = -self.impact

            prev_cash = current_cash
            current_cash += -prices.loc[date[i], symbol] * action * (1 + impact)

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
        ema_20, ema_50, macd, roc = get_discretized(sd, ed, symbol)

        prices = get_prices(sd, ed, symbol)
        portfolio = pd.DataFrame(index=prices.index, columns=[symbol])
        portfolio[:] = 0
        date = prices.index
        current_position = 0

        # train the learner
        for i in range(portfolio.shape[0] - 1):
            s_prime = get_s_prime(current_position, ema_20.loc[date[i]],
                                  ema_50.loc[date[i]], macd.loc[date[i]],
                                  roc.loc[date[i]])

            vote = self.learner.querysetstate(s_prime)
            if vote == 0:
                action = -1000 - current_position
            elif vote == 1:
                action = -current_position
            else:
                action = 1000 - current_position

            current_position += action
            portfolio.iloc[i, 0] = action

        return portfolio


""" Helper Methods """

def get_prices(sd, ed, symbol):
    syms=[symbol]
    dates = pd.date_range(sd, ed)
    df = get_data(syms, dates)
    prices = df[syms]
    prices = prices.ffill().bfill()
    spy = df[['SPY']]
    return prices


def get_discretized(sd, ed, symbol):
    prices = get_prices(sd, ed, symbol)

    ema_20 = indicators.ema(sd, ed, symbol)
    ema_50 = indicators.ema(sd, ed, symbol, window_size=50)

    ema_20 = (prices > ema_20) * 1
    ema_50 = (prices > ema_50) * 1

    macd, macd_signal = indicators.macd(sd, ed, symbol)
    macd = (macd > macd_signal) * 1

    roc = indicators.roc(sd, ed, symbol)
    roc = (roc > 0) * 1

    return ema_20, ema_50, macd, roc


def get_s_prime(current_position, ema_20, ema_50, macd, roc):
    state = 16 if current_position == 0 else 32
    state += ema_20 * 8 + ema_50 * 4 + macd * 2 + roc
    return int(state)
