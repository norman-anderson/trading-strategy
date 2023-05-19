import pandas as pd
from util import get_data, plot_data


def author():
    return "nanderson83"


def compute_portvals(orders, start_val=1000000, commission=9.95, impact=0.005,):
    # modification orders file for project 6
    # orders = pd.read_csv(orders_file)
    # orders.sort_values('Date', 0)
    # get start and end date ranges and get the stocks data frame
    start, end = orders.index[0], orders.index[-1]
    symbol = orders.columns[0]
    stocks = get_data([symbol], pd.date_range(start, end))
    stocks.drop(['SPY'], axis=1, inplace=True)

    # create portfolio data frame and add cash balance column
    portfolio = pd.DataFrame(index=stocks.index, columns=stocks.columns)
    portfolio.fillna(0, inplace=True)
    portfolio['Cash'] = 0
    portfolio['Cash'][0] = start_val

    # calculate each day changes in the portfolio
    for index, row in orders.iterrows():
        # Project 6 modification
        if row[0] != 0:
            portfolio.loc[index, symbol] += row[0]
            portfolio.loc[index, 'Cash'] -= stocks.loc[index, symbol] * (1 + impact) * row[0] + commission

    # calculate the cumulative balance for stocks and cash account daily
    for i in range(1, len(portfolio)):
        portfolio.iloc[i, :] += portfolio.iloc[i - 1, :]

    stocks['Cash'] = 1  # add a cash column to be able to multiply
    # calculate daily holding value of each stock and sum the columns
    portvals = (portfolio * stocks).sum(axis=1)
    portvals = portvals.to_frame()
    portvals = portvals.rename(columns={0: 'Cash'})
    return portvals
