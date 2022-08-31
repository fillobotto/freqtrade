import logging
import math
from ast import Num
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from operator import getitem
from random import randint
from typing import Optional, Union

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

df = pandas.read_json('data.json')
df.columns = ["date", "open", "high", "low", "close", "volume"]

df['sar'] = ta.SAR(df)
df['ema20'] = ta.EMA(df['close'], timeperiod=20)
df['ema40'] = ta.EMA(df['close'], timeperiod=40)

minute_offsets = 30 * 1440
starting_balance = 1000

start_ts = df.tail(1).iloc[-1].squeeze()
starting_ts = datetime.fromtimestamp(start_ts['date'] / 1000) - timedelta(minutes=minute_offsets)

end_ts = df.iloc[-1].squeeze()
ending_ts = datetime.fromtimestamp(end_ts['date'] / 1000)

def how_many_are(df : DataFrame, n : int):
	block = None

	while n >= 0:
		last_candle = df.iloc[-1].squeeze()
		sar_reverse_index = df['sar'].lt(df['open'])[::-1].idxmax() if last_candle['sar'] > last_candle['open'] else df['sar'].gt(df['open'])[::-1].idxmax()
		rolling_window_length = len(df) - sar_reverse_index - 1
		block = df.iloc[-rolling_window_length:]
		df = df.head(len(df) - len(block))
		n -= 1

	return block

def find_n_segment(df : DataFrame, n : int):
	block = None

	while n >= 0:
		last_candle = df.iloc[-1].squeeze()
		sar_reverse_index = df['sar'].lt(df['open'])[::-1].idxmax() if last_candle['sar'] > last_candle['open'] else df['sar'].gt(df['open'])[::-1].idxmax()
		rolling_window_length = len(df) - sar_reverse_index - 1
		block = df.iloc[-rolling_window_length:]
		df = df.head(len(df) - len(block))
		n -= 1

	return block

start_step = len(df.index) - int(minute_offsets / 5)

for i in range(start_step, len(df.index)):

	current_ts = ending_ts - timedelta(minutes=(i - start_step))

	current_head = df.head(i)

	prev_candle = current_head.iloc[-2].squeeze()
	last_candle = current_head.iloc[-1].squeeze()

	if trade_open:
		if (last_candle['sar'] > last_candle['open']):
			trades[trade_id]['close_at'] = current_ts - timedelta(hours=1)
			trades[trade_id]['rate_close'] = last_candle['open']
			prof = ((trades[trade_id]['rate_close'] - trades[trade_id]['rate_open']) / trades[trade_id]['rate_close']) - 0.0015
			trades[trade_id]['profit'] = prof
			trade_open = False

	if(last_candle['sar'] < last_candle['open'] and prev_candle['sar'] > prev_candle['close']):

		ema_good = last_candle['ema20'] > last_candle['ema40']
		sl = (last_candle['sar'] - last_candle['open']) / last_candle['sar']

		# if(sl > -0.002):
		# 	continue

		# drop last row
		current_head = current_head[:-1]

		prev_sar = find_n_segment(current_head, 0)
		prev_2_sar = find_n_segment(current_head, 2)

		prev_sar_len = len(prev_sar)
		last_prev_sar = prev_sar.iloc[-1].squeeze()
		first_prev_sar = prev_sar.iloc[0].squeeze()

		prev_2_sar_len = len(prev_2_sar)
		last_prev_2_sar = prev_2_sar.iloc[-1].squeeze()
		first_prev_2_sar = prev_2_sar.iloc[0].squeeze()

		if not trade_open:
			trade_id = randint(1000, 99999)
			trades[trade_id] = {}
			trades[trade_id]['open_at'] = current_ts - timedelta(hours=1)
			trades[trade_id]['rate_open'] = last_candle['close']
			trades[trade_id]['diff'] = 0
			trade_open = True

profit = 0.0
win = 0
loss = 0
draw = 0
duration = 0

mean_diff_win = 0
mean_diff_loss = 0

trades = OrderedDict(sorted(trades.items(), key=lambda x: float(trades[x[0]]["profit"])))

for t in trades:
	if 'close_at' not in trades[t]:
		continue
	duration += (trades[t]['close_at'] - trades[t]['open_at']).total_seconds()
	current_profit = trades[t]['profit']
	starting_balance = starting_balance * (1 + current_profit)
	profit += current_profit
	if(abs(current_profit) < 0.0025):
		draw += 1
	elif(current_profit > 0):
		# print(f"{trades[t]['diff']} {current_profit}")
		mean_diff_win += trades[t]['diff']
		win += 1
	if(current_profit < 0):
		# print(f"{trades[t]['diff']} {current_profit}")
		mean_diff_loss += trades[t]['diff']
		loss += 1

if(len(trades) > 0):
	mean_diff_win = mean_diff_win / win
	mean_diff_loss = mean_diff_loss / loss

	duration = (duration / len(trades)) / 60
	print(f"BACKTEST DURATION: {round((ending_ts - starting_ts).total_seconds() / (60 * 1440), 1)} days")
	print(f"AVG TRADE DURATION: {round(duration, 1)} minutes")
	print(f"TOTAL PROFIT: {round(profit * 100, 2)} % over {len(trades)} trades")
	print(f"FINAL BALANCE: {round(starting_balance, 2)} €")
	print(f"Win: {win} / Draw: {draw} / Loss: {loss}")
	print(f"Tan - Win: {mean_diff_win} / Loss: {mean_diff_loss}")
