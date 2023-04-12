from util import get_data, plot_data
import datetime as dt
import numpy as np
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import StrategyLearner as sl


def author():
    return "nanderson83"


if __name__ == "__main__":
    sl = sl.StrategyLearner()
