import logging
import math
import pathlib
from ast import Num
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from operator import getitem
from random import randint
from typing import List, Optional, Union

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
TIMEFRAME = '15m'
TIMEFRAME_IN_MIN = 60

#
#	PAIR
#
PAIR = 'BTC_USDT_USDT'

#
#	STARTUP DATE
#
END = datetime(2022, 7, 20, 5, 30)
END = datetime.timestamp(END) * 1000

df : DataFrame = pandas.read_json('user_data/data/okx/futures/{0}-{1}-futures.json'.format(PAIR, TIMEFRAME))
df.columns = ["date", "open", "high", "low", "close", "volume"]

# df = df[(df['date'] < END)]

#
#	INDICATORS
#
df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
df['hlc2'] = (df['high'] + df['low']) / 2

df['rsi'] = ta.RSI(df['hlc3'], timeperiod=14)
df['srsi'] = ta.EMA(df['rsi'], timeperiod=10)
df['atr'] = ta.ATR(df, timeperiod=14)

#
#   HELPER FUNCTIONS
#


class TrendType:
    UP = 1
    DOWN = 2

def is_near(p1: float, p2: float, atr: float):
    return abs(p1 - p2) < atr / 2


def reversal_condition_1(_trend: int, _price_start: float, _price_end: float):
    """
    Reversal occurs when a peak is far from the previous one (both ways)
    """
    return (_trend == TrendType.UP and _price_end > _price_start) or \
           (_trend == TrendType.DOWN and _price_end < _price_start)

def reversal_condition_2(_trend: int, _price_start: float, _price_end: float, _price_broke: float):
    """
    Reversal occurs when a peak is far from the previous one (both ways)
    """
    return (_trend == TrendType.UP and _price_end < _price_start < _price_broke) or \
           (_trend == TrendType.DOWN and _price_end > _price_start > _price_broke)

def find_reversal_1(trend: int, _df: DataFrame) -> [datetime, int]:
    # Check RSI divergence for each [RSI-oversold, RSI-overbought] segment
    reversal_found = False
    df0 = _df.copy()
    while not reversal_found:

        # Remove initial data
        peak_index = df0['rsi'].lt(70)[::-1] if trend == TrendType.DOWN else df0['rsi'].gt(30)[::-1]

        if not peak_index.empty:
            peak_index = peak_index.idxmin()
            rolling_window_length = len(df0) - peak_index - 1
            block = df0.iloc[-rolling_window_length:]
            df0 = df0.head(len(df0) - len(block))

        # Final peak
        peak_index = df0['rsi'].gt(70)[::-1] if trend == TrendType.DOWN else df0['rsi'].lt(30)[::-1]

        if peak_index.empty:
            return [None, None]

        price_end = df0.iloc[peak_index.idxmin()]
        rolling_window_length = len(df0) - peak_index.idxmin() - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 6))

        # Initial peak
        peak_index = df0['rsi'].lt(30)[::-1].idxmin() if trend == TrendType.DOWN else df0['rsi'].gt(70)[::-1].idxmin()
        price_start = df0.iloc[peak_index]
        rolling_window_length = len(df0) - peak_index - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 6))

        reversal_found = reversal_condition_1(trend, price_start['high'], price_end['high'])

        if reversal_found:
            print(
                f"Reversal #1 of {'uptrend' if trend == TrendType.DOWN else 'downtrend'} detected at {datetime.fromtimestamp(price_start['date'] / 1000.0).strftime('%d/%m/%Y, %H:%M')}")
            return [datetime.fromtimestamp(price_start['date'] / 1000.0), trend]
    return [None, None]

def find_reversal_2(trend: int, _df: DataFrame) -> [datetime, int]:
    reversal_found = False
    df0 = _df.copy()
    while not reversal_found:

        # Remove initial data
        peak_index = df0['rsi'].gt(30)[::-1] if trend == TrendType.DOWN else df0['rsi'].lt(70)[::-1]
        if not peak_index.empty:
            rolling_window_length = len(df0) - peak_index.idxmin() - 1
            block = df0.iloc[-rolling_window_length:]
            df0 = df0.head(len(df0) - len(block))

        # Final broke peak
        peak_index = df0['rsi'].lt(30)[::-1] if trend == TrendType.DOWN else df0['rsi'].gt(70)[::-1]

        if peak_index.empty:
            return [None, None]

        price_peak_broken = df0.iloc[peak_index.idxmin()]
        rolling_window_length = len(df0) - peak_index.idxmin() - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 3))

        # Remove data
        peak_index = df0['rsi'].lt(70)[::-1].idxmin() if trend == TrendType.DOWN else df0['rsi'].gt(30)[::-1].idxmin()
        rolling_window_length = len(df0) - peak_index - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 3))

        # Second peak for reversal
        peak_index = df0['rsi'].gt(70)[::-1].idxmin() if trend == TrendType.DOWN else df0['rsi'].lt(30)[::-1].idxmin()
        second_peak = df0.iloc[peak_index]
        rolling_window_length = len(df0) - peak_index - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 3))

        # First peak for reversal
        peak_index_2 = df0['rsi'].lt(70)[::-1].idxmin() if trend == TrendType.DOWN else df0['rsi'].gt(30)[::-1].idxmin()
        first_peak = df0.iloc[peak_index_2]
        rolling_window_length = len(df0) - peak_index_2 - 1
        block = df0.iloc[-rolling_window_length:]
        df0 = df0.head(len(df0) - len(block) - (4 * 3))

        inverted_peak = (_df.iloc[peak_index_2:peak_index]['rsi'].gt(50) if trend == TrendType.UP else _df.iloc[peak_index_2:peak_index]['rsi'].lt(50)).any()

        if not inverted_peak:
            continue

        reversal_found = reversal_condition_2(trend, first_peak['close'], second_peak['close'], price_peak_broken['close'])

        if reversal_found:
            print(
                f"Reversal #2 of {'uptrend' if trend == TrendType.DOWN else 'downtrend'} detected at {datetime.fromtimestamp(price_peak_broken['date'] / 1000.0).strftime('%d/%m/%Y, %H:%M')}")
            return [datetime.fromtimestamp(price_peak_broken['date'] / 1000.0), trend]
    return [None, None]

def find_reversal(_df: DataFrame):
    for i in reversed(range(1, len(_df.index))):
        head = _df.tail(i)
        current_candle = head.iloc[-1].squeeze()

        # Find peaks for reversal computing
        peaks = []
        temp_i = 4 * 4
        while len(peaks) < 3:
            current_temp_candle = head.iloc[-temp_i].squeeze()

            if current_temp_candle['rsi'] > 70 or current_temp_candle['rsi'] < 30:
                peaks.append({
                    'price': current_temp_candle['close'],
                    'atr': current_temp_candle['atr'],
                    'date': datetime.fromtimestamp(current_temp_candle['date'] / 1000.0).strftime("%d/%m/%Y, %H:%M")
                })
                # Skip 2Hs of candles to avoid taking too near peaks TODO handle better
                temp_i += 4 * 2

            temp_i += 1

            if temp_i > 4 * 24 * 4:
                return None

        # Try to determine trend
        trend = None
        if peaks[0]['price'] < peaks[1]['price'] or is_near(peaks[0]['price'], peaks[1]['price'], peaks[0]['atr']):
            trend = TrendType.DOWN
        if peaks[0]['price'] > peaks[1]['price'] or is_near(peaks[0]['price'], peaks[1]['price'], peaks[0]['atr']):
            trend = TrendType.UP

        if trend is None:
            return None

        # Find the nearest reversal
        dt1, side1 = find_reversal_1(trend, _df)
        dt2, side2 = find_reversal_2(trend, _df)

        if dt1 is None and dt2 is not None:
            return side2
        elif dt1 is not None and dt2 is None:
            return side1
        elif dt1 is not None and dt2 is not None:
            return side1 if dt1 > dt2 else dt2
        else:
            return None

def has_momentum(_df: DataFrame, side: int, rate: float) -> [bool, float]:
    df = _df.copy()
    df = df.head(len(df) - 6 * 4)

    peak_index = df['rsi'].lt(70)[::-1] if side == TrendType.DOWN else df['rsi'].gt(30)[::-1]
    if peak_index.empty:
        return [False, None]

    peak = df.iloc[peak_index.idxmin()]

    rolling_window_length = len(df) - peak_index.idxmin() - 1
    block = df.iloc[-rolling_window_length:]
    df = df.head(len(df) - len(block))

    peak_index_2 = df['rsi'].gt(70)[::-1] if side == TrendType.DOWN else df['rsi'].lt(30)[::-1]
    if not peak_index_2.empty:
        peak = df.iloc[peak_index_2.idxmin()]

    peak = peak['high' if side == TrendType.DOWN else 'low']

    counter_peak = _df.iloc[peak_index.idxmin():]['low' if side == TrendType.DOWN else 'high'].min()

    if side == TrendType.DOWN:
        return [(100 * (rate - counter_peak)) / (peak - counter_peak) > 0.35, peak]
    else:
        return [(100 * (counter_peak - rate)) / (counter_peak - peak) > 0.35, peak]


def has_momentum_down(_df: DataFrame, rate: float) -> [bool, float]:
    df = _df.copy()
    df = df.head(len(df)- 3 * 4)

    peak_index = df['rsi'].gt(70)[::-1]
    if not peak_index.empty:
        rolling_window_length = len(df) - peak_index.idxmin() - 1
        if rolling_window_length > 0:
            block = df.iloc[-rolling_window_length:]
            df = df.head(len(df) - len(block))

    peak_index_2 = df['srsi'].gt(50)[::-1]
    if not peak_index_2.empty:
        rolling_window_length = len(df) - peak_index_2.idxmin() - 1
        block = df.iloc[-rolling_window_length:]
        df = df.head(len(df) - len(block))

    peak_index_2 = df['rsi'].lt(70)[::-1]
    if peak_index_2.empty:
        return [False, None]
    peak_2 = df.iloc[peak_index_2.idxmin()]['low']
    rolling_window_length = len(df) - peak_index_2.idxmin() - 1
    block = df.iloc[-rolling_window_length:]
    df = df.head(len(df) - len(block))

    peak_index_3 = df['rsi'].gt(70)[::-1]
    if not peak_index_2.empty:
        peak_2 = _df.iloc[peak_index_3.idxmin():peak_index_2.idxmin()]['low'].max()

    return [peak_2 > rate, peak_2]


def has_momentum_up(_df: DataFrame, rate: float) -> [bool, float]:
    df = _df.copy()
    df = df.head(len(df)- 3 * 4)

    peak_index = df['rsi'].lt(30)[::-1]
    if not peak_index.empty:
        rolling_window_length = len(df) - peak_index.idxmin() - 1
        if rolling_window_length > 0:
            block = df.iloc[-rolling_window_length:]
            df = df.head(len(df) - len(block))

    peak_index_2 = df['srsi'].lt(50)[::-1]
    if not peak_index_2.empty:
        rolling_window_length = len(df) - peak_index_2.idxmin() - 1
        block = df.iloc[-rolling_window_length:]
        df = df.head(len(df) - len(block))

    peak_index_2 = df['rsi'].gt(30)[::-1]
    if peak_index_2.empty:
        return [False, None]
    peak_2 = df.iloc[peak_index_2.idxmin()]['high']
    rolling_window_length = len(df) - peak_index_2.idxmin() - 1
    block = df.iloc[-rolling_window_length:]
    df = df.head(len(df) - len(block))

    peak_index_3 = df['rsi'].lt(30)[::-1]
    if not peak_index_3.empty:
        peak_2 = _df.iloc[peak_index_3.idxmin():peak_index_2.idxmin()]['high'].max()

    return [peak_2 < rate, peak_2]

def compute_sl(sl: float, rate: float, side: int):
    if side == TrendType.UP:
        return sl if (rate - sl) / rate < 0.03 else (rate * 0.03)
    else:
        return sl if (sl - rate) / rate < 0.03 else (rate * 1.03)

#
#   MAIN LOOP
#

trade_open = False
trade_side = None
trade_tp = 0
trade_sl = 0
trade_price = None
flag_for_exit = False
profit = 0
profits = []
win = 0
lost = 0

last_successful = False
last_side = None

def compute_profit(open: float, rate: float, side: int):
    global win
    global lost
    global last_successful
    global last_side
    current_profit = (open - rate if side == TrendType.DOWN else rate - open) / rate
    last_successful = current_profit > 0
    last_side = side
    profits.append(profit)
    return current_profit

plt = df.tail(len(df.index) - 4 * 24 * 10).reset_index()['close'].plot(figsize=(80, 40), grid=True, color='cornflowerblue',
                                                      linewidth=0.5)

j = 0
for i in range(4 * 24 * 10, len(df.index)):
    current = df.head(i)
    current_candle = current.iloc[-1].squeeze()
    prev_candle = current.iloc[-2].squeeze()

    if not trade_open:
        if prev_candle['rsi'] > 70 > current_candle['rsi']:

            _has_momentum, sl = has_momentum_down(current, current_candle['close'])
            if _has_momentum:

                # Discard the first entry signal after the previous signal (of opposite side) aws successful
                if last_successful and last_side == TrendType.UP:
                    last_successful = False
                    continue

                trade_open = True
                trade_price = current_candle['close']
                trade_side = TrendType.DOWN
                trade_sl = compute_sl(sl, current_candle['close'], TrendType.DOWN)
                trade_tp = current_candle['close'] - (sl - current_candle['close'])
                plt.annotate("open S", (j, current_candle['hlc3']))
                plt.scatter(j, current_candle['close'], color='blue', s=50, alpha=0.5)
            else:
                plt.scatter(j, current_candle['close'], color='yellow', s=20, alpha=0.5)

        if prev_candle['rsi'] > 30 > current_candle['rsi']:

            _has_momentum, sl = has_momentum_up(current, current_candle['close'])
            if _has_momentum:

                # Discard the first entry signal after the previous signal (of opposite side) aws successful
                if last_successful and last_side == TrendType.DOWN:
                    last_successful = False
                    continue

                trade_open = True
                trade_price = current_candle['close']
                trade_side = TrendType.UP
                trade_sl = compute_sl(sl, current_candle['close'], TrendType.UP)
                trade_tp = current_candle['close'] + (current_candle['close'] - sl)
                plt.annotate("open L", (j, current_candle['hlc3']))
                plt.scatter(j, current_candle['close'], color='blue', s=50, alpha=0.5)
            else:
                plt.scatter(j, current_candle['close'], color='yellow', s=20, alpha=0.5)


    else:

        # Exit signal
        if (trade_side == TrendType.UP and prev_candle['rsi'] > 70 > current_candle['rsi']) or \
                (trade_side == TrendType.DOWN and prev_candle['rsi'] > 30 > current_candle['rsi']):
            plt.scatter(j, current_candle['close'], color='green', s=50, alpha=0.5)
            plt.annotate("close L (signal)" if trade_side == TrendType.UP else "close S (signal)", (j, current_candle['close']))
            profit += compute_profit(trade_price, current_candle['close'], trade_side)
            trade_open = False
            flag_for_exit = False

        #  Take profit
        # if (trade_side == TrendType.UP and current_candle['close'] > trade_tp) or \
        #         (trade_side == TrendType.DOWN and current_candle['close'] < trade_tp):
        #     plt.scatter(j, current_candle['close'], color='green', s=50, alpha=0.5)
        #     plt.annotate("close L" if trade_side == TrendType.UP else "close S", (j, current_candle['close']))
        #     trade_open = False
        #     flag_for_exit = False

        # Stop loss
        if (trade_side == TrendType.UP and current_candle['close'] < trade_sl) or \
                (trade_side == TrendType.DOWN and current_candle['close'] > trade_sl):
            plt.scatter(j, current_candle['close'], color='red', s=50, alpha=0.5)
            plt.annotate("close L" if trade_side == TrendType.UP else "close S", (j, current_candle['close']))
            profit += compute_profit(trade_price, current_candle['close'], trade_side)
            trade_open = False
            flag_for_exit = False


        if (trade_side == TrendType.UP and trade_price > current_candle['close'] and prev_candle['rsi'] > 30 > current_candle['rsi']) or \
                (trade_side == TrendType.DOWN and trade_price < current_candle['close'] and prev_candle['rsi'] > 70 > current_candle['rsi']):
            flag_for_exit = True

        # if (trade_side == TrendType.UP and flag_for_exit and prev_candle['rsi'] < 70 < current_candle['rsi']) or \
        #         (trade_side == TrendType.DOWN and flag_for_exit and prev_candle['rsi'] < 30 < current_candle['rsi']):
        #     plt.scatter(j, current_candle['close'], color='red', s=50, alpha=0.5)
        #     plt.annotate("close L (rec)" if trade_side == TrendType.UP else "close S (rec)", (j, current_candle['close']))
        #     profit += compute_profit(trade_price, current_candle['close'], trade_side)
        #     trade_open = False
        #     flag_for_exit = False

    j += 1

fig = plt.get_figure()
fig.savefig(f"{pathlib.Path(__file__).parent.absolute()}/plots/output.png")


print(profit)

# import matplotlib.pyplot as plot
#
# plot.clf()
# plot.plot(profits)
# plot.show()
