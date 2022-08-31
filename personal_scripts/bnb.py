import logging
import math
import pathlib
from ast import Num
from code import interact
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from glob import glob
from operator import getitem
from random import randint
from typing import List, Optional, Union
from xmlrpc.client import Boolean

import numpy as np
import pandas
import pandas as pd
import pytz
import talib.abstract as ta
from pandas import NA, DataFrame, Series, concat
from scipy.signal import argrelextrema
from telegram import base

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import PairLocks, Trade
from freqtrade.strategy import (DecimalParameter, IStrategy, RealParameter, informative,
                                merge_informative_pair, stoploss_from_open)


logger = logging.getLogger(__name__)

trades = {}
trade_id = None

#
#	TIMEFRAME
#
TIMEFRAME = '1h'
TIMEFRAME_IN_MIN = 60

#
#	PAIR
#
PAIR = 'BNB_USDT'

df : DataFrame = pandas.read_json('user_data/data/binance/{0}-{1}.json'.format(PAIR, TIMEFRAME))
df.columns = ["date", "open", "high", "low", "close", "volume"]

df['rsi'] = ta.RSI(df['close'], timeperiod=14)
df['ema'] = ta.EMA(df['close'], timeperiod=40)
df['fastk'], df['fastd'] = ta.STOCH(df['rsi'], df['rsi'], df['rsi'], 28)

START = datetime(2022, 5, 10, 5, 0)
START = datetime.timestamp(START) * 1000

END = datetime(2022, 8, 14, 18, 0)
END = datetime.timestamp(END) * 1000

df = df[(df['date'] < END)]
df = df[(df['date'] > START)]

# Buy signal
# crossed_above(df['fastk'], 5)

df['time'] = df['date'].apply(lambda x: datetime.fromtimestamp(x / 1000).strftime("%d/%m, %H:%M:%S"))

df['overbought'] = ((df['fastk'] > 90)).astype(bool)
df['oversold'] = ((df['fastk'] < 20)).astype(bool)

def confirm_entry(_df : DataFrame):
    current_candle = _df.iloc[-1].squeeze()

    stoch_overbought = _df.loc[(_df['overbought'].astype(Boolean) == False)[::-1].idxmin():].iloc[0]
    stoch_overbought_index = _df.loc[(_df['overbought'].astype(Boolean) == False)[::-1].idxmin():].index.values[0]

    _df = _df.loc[(_df.index < stoch_overbought_index)]
    stoch_oversold = _df.loc[(_df['oversold'].astype(Boolean) == False)[::-1].idxmin():].iloc[0]
    stoch_oversold_index = _df.loc[(_df['oversold'].astype(Boolean) == False)[::-1].idxmin():].index.values[0]

    oversold_diff = abs((stoch_oversold['close'] - stoch_oversold['ema']) / stoch_oversold['close'])
    overbought_diff = abs((stoch_overbought['close'] - stoch_overbought['ema']) / stoch_overbought['close'])

    return [
        (
                current_candle['close'] > stoch_oversold['close'] and
                stoch_overbought['high'] > stoch_overbought['ema']
        )
        ,
        stoch_oversold['high'],
        abs(oversold_diff - overbought_diff)
    ]

for i in range(28, len(df.index)):
    current = df.head(i)
    current_candle = current.iloc[-1].squeeze()
    prev_candle = current.iloc[-2].squeeze()

    if prev_candle['fastk'] < 5 and current_candle['fastk'] > 5:
        confirm_signal, stop_loss, volatility = confirm_entry(current)
        if confirm_signal and volatility > 0.01:
            print(f"BUY @ {current_candle['time']} - vol: {volatility}")
