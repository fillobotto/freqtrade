import logging
import math
from ast import Num
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from operator import getitem
from random import randint
from typing import List, Optional, Union

import numpy as np
import pandas
import pytz
import talib.abstract as ta
from pandas import NA, DataFrame, Series
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
TIMEFRAME = '1h'
TIMEFRAME_IN_MIN = 60

#
#	PAIR
#
PAIR = 'SOL_USDT'

#
#	STARTUP DATE
#
START = datetime(2022, 1, 10)


df = pandas.read_json('user_data/data/binance/{0}-{1}.json'.format(PAIR, TIMEFRAME))
df.columns = ["date", "open", "high", "low", "close", "volume"]

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

def is_support(df : DataFrame):
	support = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2)) & (df['low'].shift(2) < df['low'].shift(3)) & (df['low'].shift(3) < df['low'].shift(4))
	return (support.iloc[-1], df['low'].shift(2).iloc[-1])

def is_resistance(df : DataFrame):
	resistance = (df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2)) & (df['high'].shift(2) > df['high'].shift(3)) & (df['high'].shift(3) > df['high'].shift(4))
	return (resistance.iloc[-1], df['high'].shift(2).iloc[-1])

def is_far_from_level(current, level, volatility):
	if(level is None):
		return True
	return abs(current - float(level)) > volatility

def get_price_volatility(df : DataFrame):
	return np.mean(df['high'] - df['low']) * 2

def get_near_level(current, levels : List, volatility):
	if(len([(a, x) for (a, x) in enumerate(levels) if 'price' in x and abs(current - float(x['price'])) < volatility]) == 0):
		return (-1, None)
	return [(a, x) for (a, x) in enumerate(levels) if abs(current - float(x['price'])) < volatility][0]

def clean_levels(levels : List):
	l0 = []
	for (idx, x) in enumerate(levels):
		if(x['hits'] >= WIDE_RANGE_HIT_THRESHOLD):
			l0.append(x)
	return l0

start_ts = df.tail(1).iloc[-1].squeeze()
st_of =  datetime.fromtimestamp(start_ts['date'] / 1000)

minute_offsets = ((st_of - START).total_seconds() / 60)

starting_ts = datetime.fromtimestamp(start_ts['date'] / 1000) - timedelta(minutes=minute_offsets)

end_ts = df.iloc[-1].squeeze()
ending_ts = datetime.fromtimestamp(end_ts['date'] / 1000)

start_step = len(df.index) - int(minute_offsets / TIMEFRAME_IN_MIN)

for i in range(start_step, len(df.index)):

	current_ts = ending_ts + timedelta(minutes=(i - start_step)) - timedelta(minutes=minute_offsets)

	current_head = df.head(i)

	volatility = get_price_volatility(current_head)
	is_support_x, price_support = is_support(current_head)

	if(is_support_x):
		idx, lvls = get_near_level(price_support, supports, volatility * 2)
		if(lvls is None):
			supports.append({ 'price': price_support, 'hits': 0 })
		else:
			supports[idx]['hits'] += 1

	is_resistance_x, price_resistance = is_resistance(current_head)

	if(is_resistance_x):
		idx, lvls = get_near_level(price_resistance, resistances, volatility * 2)
		if(lvls is None):
			resistances.append({ 'price': price_resistance, 'hits': 0 })
		else:
			resistances[idx]['hits'] += 1

supports = clean_levels(supports)
resistances = clean_levels(resistances)

print(resistances)
print(supports)
