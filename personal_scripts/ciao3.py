import logging
import math
from ast import Num
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from operator import getitem
from random import randint
from typing import List, Optional, Union

import numpy as np
import pandas
import pytz
import talib.abstract as ta
from pandas import NA, DataFrame, Series, concat
from scipy.signal import argrelextrema
from telegram import base

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import PairLocks, Trade
from freqtrade.strategy import (DecimalParameter, IStrategy, RealParameter, informative,
                                merge_informative_pair, stoploss_from_open)
from freqtrade.strategy.hyper import IntParameter


logger = logging.getLogger(__name__)

trades = {}
trade_open = False
trade_id = None

#
#	TIMEFRAME
#
TIMEFRAME = '15m'
TIMEFRAME_IN_MIN = 60

#
#	PAIR
#
PAIR = 'BTC_USDT_USDT'

#
#	STARTUP DATE
#
START = datetime(2022, 1, 10)


df : DataFrame = pandas.read_json('user_data/data/okx/futures/{0}-{1}-futures.json'.format(PAIR, TIMEFRAME))
df.columns = ["date", "open", "high", "low", "close", "volume"]
df.tail(24 * 30 * 4)
#
#	INDICATORS
#

#
#	END INDICATORS
#

#
#	OUTPUT VARS
#
WIDE_RANGE_HIT_THRESHOLD = 5
supports = []
resistances = []
#
#	END OUTPUT VARS
#

def get_max_min(prices : DataFrame, smoothing, window_range):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_max_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_min_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmin())
    maxima = DataFrame(prices.loc[price_local_max_dt])
    minima = DataFrame(prices.loc[price_local_min_dt])
    max_min = concat([maxima, minima]).sort_index()
    #max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['date'] = p[p['date'].isin(max_min.date)].values
    max_min = max_min.set_index('date')['close']

    return max_min


minmax = get_max_min(df, 15, 20)

plt = df.reset_index()['close'].plot(figsize=(100,40),grid=True)

plt.scatter(minmax.index, minmax.values, color='orange', alpha=0.5)

# def find_patterns(max_min):
#     patterns = defaultdict(list)
#
#     # Window range is 5 units
#     for i in range(5, len(max_min)):
#         window = max_min.iloc[i - 5:i]
#
#         # Pattern must play out in less than n units
#         if window.index[-1] - window.index[0] > 100:
#             continue
#
#         a, b, c, d, e = window.iloc[0:5]
#
#         # IHS
#         if a < b and c < a and c < e and c < d and e < d and abs(b - d) <= np.mean([b, d]) * 0.02 and abs(a - e) <= np.mean([a, e]) * 0.02 \
#                 and np.mean([abs(a -b), abs(e - d)]) <= np.mean([abs(a - c), abs(e - c)]):
#             patterns['IHS'].append((window.index[0], window.index[-1]))
#
#     return patterns
#
#
# patterns = find_patterns(minmax)
#
# for name, end_day_nums in patterns.items():
#     for i, tup in enumerate(end_day_nums):
#         sd = tup[0]
#         ed = tup[1]
#         plt.scatter(minmax.loc[sd:ed].index,
#                       minmax.loc[sd:ed].values,
#                       s=200, alpha=.3)



uptrend = True
for i in range(1, len(minmax.index.values)):
    row = minmax.take([-i])
    price = row.values[0]
    index = row.index[0]

    if uptrend:
        if minmax.take([-i - 1]).values[0] < price:
            pass
        else:
            temp_i = 1
            major_count = 0
            while abs(-i - temp_i) < len(minmax.index.values) and minmax.take([-i - temp_i]).values[0] > price:
                temp_i += 1
                major_count += 1

            if major_count > 7:
                plt.scatter(index, price, color='green' if uptrend else 'red', s=200, alpha=.3)
                uptrend = False
    else:
        if minmax.take([-i - 1]).values[0] > price:
            pass
        else:
            temp_i = 1
            major_count = 0
            while abs(-i - temp_i) < len(minmax.index.values) and minmax.take([-i - temp_i]).values[0] < price:
                temp_i += 1
                major_count += 1

            if major_count > 4:
                plt.scatter(index, price, color='green' if uptrend else 'red', s=200, alpha=.3)
                uptrend = True

    # s, r = get_levels_for_index(i)
    # if s is not None and r is not None:
    #     in_range = 'green' if (s < minmax.loc[i] < r) else 'red'
    #     plt.scatter(i,minmax.loc[i], color=in_range, s=200, alpha=.3)















def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

pairs = np.array(list(zip(minmax.index.values, minmax.values)))
from matplotlib import pyplot
# affinity propagation clustering
from numpy import unique, where
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_classification


X, _ = make_classification(n_samples=1000, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=4)

# define the model
model = AffinityPropagation(damping=0.9)

# fit the model
model.fit(pairs)
# assign a cluster to each example
yhat = model.predict(pairs)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    if row_ix:

        data = []
        points = []
        for p in row_ix[0]:
            data.append([minmax.iloc[[p.astype(np.int32)]].index.values[0], minmax.iloc[[p.astype(np.int32)]].values[0]])
            points.append(minmax.iloc[[p.astype(np.int32)]].index.values[0])
        df0 = DataFrame(data, columns=['date', 'close'])

        clusters = []
        eps = 100
        points_sorted = sorted(points)
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)

        for c in clusters:
            if len(c) > 4:
                values_in_cluster = []
                for val in c:
                    values_in_cluster.append(minmax.loc[val])
                values_in_cluster = sorted(values_in_cluster)
                values_in_cluster = np.array(values_in_cluster)
                values_in_cluster = reject_outliers(values_in_cluster, 3)

                plt.plot([np.min(c), np.max(c)], [np.min(values_in_cluster), np.min(values_in_cluster)], color='red')
                plt.plot([np.min(c), np.max(c)], [np.max(values_in_cluster), np.max(values_in_cluster)], color='green')

                supports.append({
                    'price': np.min(values_in_cluster),
                    'start': np.min(c),
                    'end': np.max(c)
                })

                resistances.append({
                    'price': np.max(values_in_cluster),
                    'start': np.min(c),
                    'end': np.max(c)
                })
                # plt.plot([np.min(c), np.max(c)], [np.mean(np.array_split(values_in_cluster, 4)[0]), np.mean(np.array_split(values_in_cluster, 4)[0])], color='red')
                # plt.plot([np.min(c), np.max(c)], [np.mean(np.array_split(values_in_cluster, 4)[-1]),np.mean(np.array_split(values_in_cluster, 4)[-1])], color='green')

        # plt.scatter(df0['date'], df0['close'])


# Check if current price is between latest support/resistance range

# current_support = supports[-1]
# current_resistance = resistances[-1]
#
# def get_levels_for_index(idx : int) -> [int, int]:
#     for i in reversed(range(0, len(supports))):
#         s = supports[i]
#         if s['start'] < idx and s['end'] > idx:
#             return [supports[i]['price'], resistances[i]['price']]
#     return [None, None]
#
# for i in minmax.index:
#     s, r = get_levels_for_index(i)
#     if s is not None and r is not None:
#         in_range = 'green' if (s < minmax.loc[i] < r) else 'red'
#         plt.scatter(i,minmax.loc[i], color=in_range, s=200, alpha=.3)



last_price = df.iloc[-1]['close']
print(last_price)


import matplotlib.pyplot as plot


plot.show()
