# coding: UTF-8
import os
import random

import math
import re
import time

import numpy
from hyperopt import hp

from src import (
    highest,
    lowest,
    avg_price,
    typ_price,
    sma,
    crossover,
    crossunder,
    over,
    under,
    last,
    rci,
    rsi,
    double_ema,
    ema,
    triple_ema,
    wma,
    ssma,
    hull,
    logger,
    notify,
    atr,
    willr,
    bbands,
    supertrend,
    heikinashi,
)
from src.bitmex import BitMex
from src.binance_futures import BinanceFutures
from src.bitmex_stub import BitMexStub
from src.binance_futures_stub import BinanceFuturesStub
from src.bot import Bot
from dotenv import load_dotenv

# Channel breakout strategy
from src.gmail_sub import GmailSub

load_dotenv()


def get_calc_lot(lot, decimal_num: int, leverage: float, actual_leverage: float):
    calc_lot = lot / leverage
    calc_lot *= actual_leverage
    calc_lot -= calc_lot % (10 ** -decimal_num)
    calc_lot = round(calc_lot, decimal_num)
    return calc_lot


def calc_entry_price(price, long, price_decimals=2):
    if long:
        return round(price - (0.1 / 100 * price), price_decimals)
    else:
        return round(price + (0.1 / 100 * price), price_decimals)


class Doten(Bot):
    def __init__(self):
        Bot.__init__(self, "2h")

    def options(self):
        return {
            "length": hp.randint("length", 1, 30, 1),
        }

    def strategy(self, open, close, high, low, volume):
        lot = self.exchange.get_lot()
        length = self.input("length", int, 9)
        up = last(highest(high, length))
        dn = last(lowest(low, length))
        self.exchange.plot("up", up, "b")
        self.exchange.plot("dn", dn, "r")
        self.exchange.entry("Long", True, round(lot / 2), stop=up)
        self.exchange.entry("Short", False, round(lot / 2), stop=dn)


# SMA CrossOver
class SMA(Bot):
    def __init__(self):
        Bot.__init__(self, "2h")

    def options(self):
        return {
            "fast_len": hp.quniform("fast_len", 1, 30, 1),
            "slow_len": hp.quniform("slow_len", 1, 30, 1),
        }

    def strategy(self, open, close, high, low, volume):
        lot = self.exchange.get_lot()
        fast_len = self.input("fast_len", int, 9)
        slow_len = self.input("slow_len", int, 16)
        fast_sma = sma(close, fast_len)
        slow_sma = sma(close, slow_len)
        golden_cross = crossover(fast_sma, slow_sma)
        dead_cross = crossunder(fast_sma, slow_sma)
        if golden_cross:
            self.exchange.entry("Long", True, lot)
        if dead_cross:
            self.exchange.entry("Short", False, lot)


# Rci
class Rci(Bot):
    def __init__(self):
        Bot.__init__(self, "5m")

    def options(self):
        return {
            "rcv_short_len": hp.quniform("rcv_short_len", 1, 10, 1),
            "rcv_medium_len": hp.quniform("rcv_medium_len", 5, 15, 1),
            "rcv_long_len": hp.quniform("rcv_long_len", 10, 20, 1),
        }

    def strategy(self, open, close, high, low, volume):
        lot = self.exchange.get_lot()

        itv_s = self.input("rcv_short_len", int, 5)
        itv_m = self.input("rcv_medium_len", int, 9)
        itv_l = self.input("rcv_long_len", int, 15)

        rci_s = rci(close, itv_s)
        rci_m = rci(close, itv_m)
        rci_l = rci(close, itv_l)

        long = ((-80 > rci_s[-1] > rci_s[-2]) or (-82 > rci_m[-1] > rci_m[-2])) and (rci_l[-1] < -10 and rci_l[-2] > rci_l[-2])
        short = ((80 < rci_s[-1] < rci_s[-2]) or (rci_m[-1] < -82 and rci_m[-1] < rci_m[-2])) and (10 < rci_l[-1] < rci_l[-2])
        close_all = 80 < rci_m[-1] < rci_m[-2] or -80 > rci_m[-1] > rci_m[-2]

        if long:
            self.exchange.entry("Long", True, lot)
        elif short:
            self.exchange.entry("Short", False, lot)
        elif close_all:
            self.exchange.close_all()


# OCC
class OCC(Bot):
    variants = [sma, ema, double_ema, triple_ema, wma, ssma, hull]
    eval_time = None

    def __init__(self):
        Bot.__init__(self, "1m")

    def ohlcv_len(self):
        return 15 * 30

    def options(self):
        return {
            "variant_type": hp.quniform("variant_type", 0, len(self.variants) - 1, 1),
            "basis_len": hp.quniform("basis_len", 1, 30, 1),
            "resolution": hp.quniform("resolution", 1, 10, 1),
            "sma_len": hp.quniform("sma_len", 1, 15, 1),
            "div_threshold": hp.quniform("div_threshold", 1, 6, 0.1),
        }

    def strategy(self, open, close, high, low, volume):
        lot = self.exchange.get_lot()

        variant_type = self.input(defval=5, title="variant_type", type=int)
        basis_len = self.input(defval=19, title="basis_len", type=int)
        resolution = self.input(defval=2, title="resolution", type=int)
        sma_len = self.input(defval=9, title="sma_len", type=int)
        div_threshold = self.input(defval=3.0, title="div_threshold", type=float)

        source = self.exchange.security(str(resolution) + "m")

        if self.eval_time is not None and self.eval_time == source.iloc[-1].name:
            return

        series_open = source["open"].values
        series_close = source["close"].values

        variant = self.variants[variant_type]

        val_open = variant(series_open, basis_len)
        val_close = variant(series_close, basis_len)

        if val_open[-1] > val_close[-1]:
            high_val = val_open[-1]
            low_val = val_close[-1]
        else:
            high_val = val_close[-1]
            low_val = val_open[-1]

        sma_val = sma(close, sma_len)
        logger.info("lagging log")
        self.exchange.plot("val_open", val_open[-1], "b")
        self.exchange.plot("val_close", val_close[-1], "r")

        self.exchange.entry("Long", True, lot, stop=math.floor(low_val), when=(sma_val[-1] < low_val))
        self.exchange.entry("Short", False, lot, stop=math.ceil(high_val), when=(sma_val[-1] > high_val))

        open_close_div = sma(numpy.abs(val_open - val_close), sma_len)

        if open_close_div[-1] > div_threshold and open_close_div[-2] > div_threshold < open_close_div[-2]:
            self.exchange.close_all()

        self.eval_time = source.iloc[-1].name


# TradingView


class TV(Bot):
    subscriber = None

    def __init__(self):
        Bot.__init__(self, "1m")

        user_id = os.environ.get("GMAIL_ADDRESS")
        if user_id is None:
            raise Exception("Please set GMAIL_ADDRESS into env to use Trading View Strategy.")
        self.subscriber = GmailSub(user_id)
        self.subscriber.set_from_address("noreply@tradingview.com")

    def __on_message(self, messages):
        for message in messages:
            if "payload" not in message:
                continue
            if "headers" not in message["payload"]:
                continue
            subject_list = [header["value"] for header in message["payload"]["headers"] if header["name"] == "Subject"]
            if len(subject_list) == 0:
                continue
            subject = subject_list[0]
            if subject.startswith("TradingViewアラート:"):
                action = subject.replace("TradingViewアラート:", "")
                self.__action(action)

    def __action(self, action):
        lot = self.exchange.get_lot()
        if re.search("buy", action, re.IGNORECASE):
            self.exchange.entry("Long", True, lot)
        elif re.search("sell", action, re.IGNORECASE):
            self.exchange.entry("Short", True, lot)
        elif re.search("exit", action, re.IGNORECASE):
            self.exchange.close_all()

    def run(self):
        if self.hyperopt:
            raise Exception("Trading View Strategy dose not support hyperopt Mode.")
        elif self.back_test:
            raise Exception("Trading View Strategy dose not support backtest Mode.")
        elif self.stub_test:
            # if you want to use binance futures
            # self.exchange = BinanceFuturesStub(account=self.account, pair=self.pair)
            self.exchange = BitMexStub(account=self.account, pair=self.pair)
            logger.info(f"Bot Mode : Stub")
        else:
            # if you want to use binance
            # self.exchange = BinanceFutures(account=self.account, pair=self.pair, demo=self.test_net)
            self.exchange = BitMex(account=self.account, pair=self.pair, demo=self.test_net)
            logger.info(f"Bot Mode : Trade")

        logger.info(f"Starting Bot")
        logger.info(f"Strategy : {type(self).__name__}")
        logger.info(f"Balance : {self.exchange.get_balance()}")

        notify(f"Starting Bot\n" f"Strategy : {type(self).__name__}\n" f"Balance : {self.exchange.get_balance()/100000000} XBT")

        self.subscriber.on_message(self.__on_message)

    def stop(self):
        self.subscriber.stop()


# candle tester


class CandleTester(Bot):
    def __init__(self):
        Bot.__init__(self, "1m")

    # this is for parameter optimization in hyperopt mode
    def options(self):
        return {}

    def strategy(self, open, close, high, low, volume):
        logger.info(f"open: {open[-1]}")
        logger.info(f"high: {high[-1]}")
        logger.info(f"low: {low[-1]}")
        logger.info(f"close: {close[-1]}")
        logger.info(f"volume: {volume[-1]}")


# sample strategy


class Sample(Bot):
    # set variables
    long_entry_signal_history = []
    short_entry_signal_history = []

    def __init__(self):
        # set time frame here
        Bot.__init__(self, "1m")

    def options(self):
        return {}

    def round_decimals_down(self, number: float, decimals: int = 2):
        """
        Returns a value rounded down to a specific number of decimal places.
        """
        if not isinstance(decimals, int):
            raise TypeError("decimal places must be an integer")
        elif decimals < 0:
            raise ValueError("decimal places has to be 0 or more")
        elif decimals == 0:
            return math.floor(number)

        factor = 10 ** decimals
        return math.floor(number * factor) / factor

    def strategy(self, open, close, high, low, volume):

        # get lot or set your own value which will be used to size orders
        # careful default lot is about 20x your account size !!!
        lot = self.exchange.get_lot()
        pos_size = self.exchange.get_position_size()

        # indicator lengths
        fast_len = self.input("fast_len", int, 6)
        slow_len = self.input("slow_len", int, 18)

        # setting indicators, they usually take source and length as arguments
        sma1 = sma(close, fast_len)
        sma2 = sma(close, slow_len)

        # entry conditions
        long_entry_condition = crossover(sma1, sma2)
        short_entry_condition = crossunder(sma1, sma2)

        # setting a simple stop loss and profit target in % using built-in simple profit take and stop loss implementation
        # which is placing the sl and tp automatically after entering a position
        self.exchange.sltp(
            profit_long=1.25,
            profit_short=1.25,
            stop_long=1,
            stop_short=1.1,
            round_decimals=0,
        )

        # example of calculation of stop loss price 0.8% round on 2 decimals hardcoded inside this class
        # sl_long = round(close[-1] - close[-1]*0.8/100, 2)
        # sl_short = round(close[-1] - close[-1]*0.8/100, 2)

        # order execution logic
        if pos_size == 0:
            if long_entry_condition:
                # entry - True means long for every other order other than entry use self.exchange.order() function
                self.exchange.entry("Long", True, self.round_decimals_down(lot / 20, 3))
                # stop loss hardcoded inside this class
                # self.exchange.order("SLLong", False, lot/20, stop=sl_long, reduce_only=True, when=False)

            if short_entry_condition:
                # entry - False means short for every other order other than entry use self.exchange.order() function
                self.exchange.entry("Short", False, self.round_decimals_down(lot / 20, 3))
                # stop loss hardcoded inside this class
                # self.exchange.order("SLShort", True, lot/20, stop=sl_short, reduce_only=True, when=False)

        # storing history for entry signals, you can store any variable this way to keep historical values
        self.long_entry_signal_history.append(long_entry_condition)
        self.short_entry_signal_history.append(short_entry_condition)

        # OHLCV and indicator data, you can access history using list index
        # log indicator values
        logger.info(f"sma1: {sma1[-1]}")
        logger.info(f"second last sma2: {sma2[-2]}")
        # log last candle OHLCV values
        logger.info(f"open: {open[-1]}")
        logger.info(f"high: {high[-1]}")
        logger.info(f"low: {low[-1]}")
        logger.info(f"close: {close[-1]}")
        logger.info(f"volume: {volume[-1]}")
        # second last candle OHLCV values
        logger.info(f"second last open: {open[-2]}")
        logger.info(f"second last high: {high[-2]}")
        logger.info(f"second last low: {low[-2]}")
        logger.info(f"second last close: {close[-2]}")
        logger.info(f"second last volume: {volume[-2]}")
        # logger.info(f"position: {pos_size}")
        # log history entry signals
        # logger.info(f"long_entry_hist: {self.long_entry_signal_history}")
        # logger.info(f"short_entry_hist: {self.short_entry_signal_history}")


# SMA CrossOver


class SMA2(Bot):
    decimal_num = int(os.environ.get("BOT_DECIMAL_NUM", 3))
    price_decimal_num = int(os.environ.get("BOT_PRICE_DECIMAL_NUM", 2))
    rr_ratio = 2
    risk = 0.5

    def __init__(self):
        Bot.__init__(self, "1m")

    def options(self):
        return {
            "fast_len": hp.quniform("fast_len", 1, 20, 1),
            "slow_len": hp.quniform("slow_len", 1, 60, 1),
            "trend_len": hp.quniform("trend_len", 1, 99, 1),
        }

    def strategy(self, open, close, high, low, volume):

        lot = self.exchange.get_lot()
        lot = get_calc_lot(lot=lot, decimal_num=self.decimal_num, leverage=20.0, actual_leverage=3.0)

        fast_len = self.input("fast_len", int, int(os.environ.get("BOT_FAST_LEN", 5)))
        slow_len = self.input("slow_len", int, int(os.environ.get("BOT_SLOW_LEN", 18)))
        trend_len = self.input("trend_len", int, 90)

        logger.info(f"fast_len: {fast_len}")
        logger.info(f"slow_len: {slow_len}")
        logger.info(f"trend_len: {trend_len}")

        fast_sma = sma(close, fast_len)
        slow_sma = sma(close, slow_len)
        trend_sma = sma(close, trend_len)

        uptrend = True if trend_sma[-1] > trend_sma[-3] or trend_sma[-1] > trend_sma[-10] else False
        downtrend = True if trend_sma[-1] < trend_sma[-3] or trend_sma[-1] < trend_sma[-10] else False

        golden_cross = crossover(fast_sma, slow_sma)
        dead_cross = crossunder(fast_sma, slow_sma)
        # inc_trend = fast_sma[-1] > slow_sma[-1]
        # dec_trend = fast_sma[-1] < slow_sma[-1]

        reward = self.risk * self.rr_ratio
        self.exchange.sltp(
            profit_long=reward,
            profit_short=reward,
            stop_long=self.risk,
            stop_short=self.risk,
            round_decimals=self.price_decimal_num,
        )

        # if float(self.exchange.get_position()['notional']) == 0.0:
        if self.exchange.get_position_size() == 0.0:

            self.exchange.cancel_all()

            if golden_cross:
                print("inc_trend detected")
                while True:
                    # check if in long position
                    if float(self.exchange.get_position()["notional"]) > 0.0 or downtrend:
                        print("long position opened")
                        break
                    print("trying to open long position...")
                    self.exchange.entry("Long", True, lot)

            if dead_cross:
                print("dec_trend detected")
                while True:
                    # check if in short position
                    if float(self.exchange.get_position()["notional"]) < 0.0 or uptrend:
                        print("short position opened")
                        break
                    print("trying to open short position...")
                    self.exchange.entry("Short", False, lot)

        # OHLCV and indicator data, you can access history using list index
        # log indicator values
        print()
        logger.info(f"fast_sma: {fast_sma[-1]}")
        logger.info(f"slow_sma: {slow_sma[-1]}")
        logger.info(f"trend_sma: {trend_sma[-1]}")
        logger.info(f"uptrend: {uptrend}")
        logger.info(f"downtrend: {downtrend}")
        logger.info(f"golden_cross: {golden_cross}")
        logger.info(f"dead_cross: {dead_cross}")
        # log last candle OHLCV values
        logger.info(f"open: {open[-1]}")
        logger.info(f"high: {high[-1]}")
        logger.info(f"low: {low[-1]}")
        logger.info(f"close: {close[-1]}")
        logger.info(f"volume: {volume[-1]}")
        # second last candle OHLCV values


class YYY(Bot):
    decimal_num = int(os.environ.get("BOT_DECIMAL_NUM", 3))
    price_decimal_num = int(os.environ.get("BOT_PRICE_DECIMAL_NUM", 2))

    def __init__(self):
        Bot.__init__(self, "1m")

    def ohlcv_len(self):
        return int(os.environ.get("BOT_TREND_LEN", 1200)) + 10

    def strategy(self, open, close, high, low, volume):
        lot = self.exchange.get_lot()
        lot = int(round(lot / 6, self.decimal_num))

        price = self.exchange.get_market_price()
        pos_size = self.exchange.get_position_size()

        fast_len = self.input("fast_len", int, int(os.environ.get("BOT_FAST_LEN", 5)))
        slow_len = self.input("slow_len", int, int(os.environ.get("BOT_SLOW_LEN", 18)))
        trend_len = self.input("trend_len", int, int(os.environ.get("BOT_TREND_LEN", 1200)))

        fast_sma = sma(close, fast_len)
        slow_sma = sma(close, slow_len)
        trend_sma = sma(close, trend_len)

        uptrend = True if trend_sma[-1] > trend_sma[-3] or trend_sma[-1] > trend_sma[-10] else False
        downtrend = True if trend_sma[-1] < trend_sma[-3] or trend_sma[-1] < trend_sma[-10] else False

        golden_cross = crossover(fast_sma, slow_sma)
        dead_cross = crossunder(fast_sma, slow_sma)

        nc = "golden" if round(fast_sma[-1] - slow_sma[-1], self.price_decimal_num) < 0 else "dead"
        ct = "sideways" if downtrend and uptrend else ("down" if downtrend else "up")

        np = "short" if nc == "golden" and (pos_size > 0 or (pos_size >= 0 and downtrend)) else "long"
        nt = "golden" if (nc == "golden" and np == "short") else ("dead" if nc == "dead" and np == "long" else not nc)

        logger.info(f"--------------------------------------")
        logger.info(f"trend: {ct}")
        logger.info(f'next trade @ {nt} cross > {np} {lot} @ {calc_entry_price(price, False, self.price_decimal_num) if np == "short" else calc_entry_price(price, True, self.price_decimal_num)}')
        if trend_sma[-1] != trend_sma[-1] or trend_sma[-3] != trend_sma[-3] or trend_sma[-10] != trend_sma[-10]:
            logger.info(f"--------------------------------------")
            logger.info(f"Bot status: NEEDS RESTART")
        # logger.info(f'--------------------------------------')
        # logger.info(f'{abs(pos_size)}')

        if not eval(os.environ.get("BOT_TEST", "False")):
            if dead_cross and uptrend:
                # self.exchange.cancel_orders_by_side('BUY')
                self.exchange.order(
                    "Long",
                    True,
                    lot,
                    limit=calc_entry_price(price, True, self.price_decimal_num),
                    when=True,
                    post_only=True,
                )
                logger.info("in dead_cross and uptrend for long")

            if float(self.exchange.get_position()["notional"]) > 0.0:
                self.exchange.order(
                    "Long",
                    False,
                    lot,
                    limit=calc_entry_price(price, False, self.price_decimal_num),
                    when=golden_cross,
                    post_only=True,
                )

            if golden_cross and downtrend:
                # self.exchange.cancel_orders_by_side('SELL')
                self.exchange.entry(
                    "Short",
                    False,
                    lot,
                    limit=calc_entry_price(price, False, self.price_decimal_num),
                    when=True,
                    post_only=True,
                )
                logger.info("in golden_cross and downtrend for short")

            if float(self.exchange.get_position()["notional"]) < 0.0:
                self.exchange.order(
                    "Short",
                    True,
                    lot,
                    limit=calc_entry_price(price, True, self.price_decimal_num),
                    stop=(calc_entry_price(price, True, self.price_decimal_num)),
                    when=dead_cross,
                    post_only=True,
                )


class Heikinashi(Bot):
    variants = [sma, ema, double_ema, triple_ema, wma, ssma, hull, heikinashi]
    eval_time = None

    def __init__(self):
        Bot.__init__(self, "1m")

    def options(self):
        return {
            "fast_len": hp.quniform("fast_len", 1, 60, 1),
            "slow_len": hp.quniform("slow_len", 1, 240, 1),
        }

    def strategy(self, open, close, high, low, volume):

        lot = self.exchange.get_lot()
        lot = int(round(lot / 6, self.decimal_num))

        resolution = self.input(defval=1, title="resolution", type=int)
        variant_type = self.input(defval=5, title="variant_type", type=int)
        basis_len = self.input(defval=19, title="basis_len", type=int)

        fast_len = self.input("fast_len", int, 1)
        slow_len = self.input("slow_len", int, 30)
        trend_len = self.input("slow_len", int, 60)
        longtrend_len = self.input("slow_len", int, 120)

        source = self.exchange.security(str(resolution) + "m")

        hadf = heikinashi(source)
        hadf_fast = heikinashi(hadf)

        ha_open_values = hadf_fast["HA_open"].values
        ha_close_values = hadf_fast["HA_close"].values
        variant = self.variants[variant_type]

        ha_open_fast = variant(ha_open_values, fast_len)
        ha_close_fast = variant(ha_close_values, fast_len)
        haopen_fast = ha_open_fast[-1]
        haclose_fast = ha_close_fast[-1]
        haup_fast = haclose_fast > haopen_fast
        hadown_fast = haclose_fast <= haopen_fast
        # logger.info('haup_fast:%s\n' % haup_fast)

        ha_open_slow = variant(ha_open_values, slow_len)
        ha_close_slow = variant(ha_close_values, slow_len)
        haopen_slow = ha_open_slow[-1]
        haclose_slow = ha_close_slow[-1]
        haup_slow = haclose_slow > haopen_slow
        hadown_slow = haclose_slow <= haopen_slow
        # logger.info('haup_slow:%s\n' % haup_slow)

        ha_open_trend = variant(ha_open_values, trend_len)
        ha_close_trend = variant(ha_close_values, trend_len)
        haopen_trend = ha_open_trend[-1]
        haclose_trend = ha_close_trend[-1]
        haup_trend = haclose_trend > haopen_trend
        hadown_trend = haclose_trend <= haopen_trend
        # logger.info('haup_trend:%s\n' % haup_trend)

        ha_open_longtrend = variant(ha_open_values, longtrend_len)
        ha_close_longtrend = variant(ha_close_values, longtrend_len)
        haopen_longtrend = ha_open_longtrend[-1]
        haclose_longtrend = ha_close_longtrend[-1]
        haup_longtrend = haclose_longtrend > haopen_longtrend
        hadown_longtrend = haclose_longtrend <= haopen_longtrend
        logger.info("ha_close_longtrend:%s\n" % ha_close_longtrend)
        logger.info("ha_open_longtrend:%s\n" % ha_open_longtrend)

        if not eval(os.environ.get("BOT_TEST", "False")):
            "long"
            self.exchange.entry("Long", True, lot, when=crossover(ha_close_longtrend, ha_open_longtrend))
            " short "
            self.exchange.entry(
                "Short",
                False,
                lot,
                when=crossunder(ha_close_longtrend, ha_open_longtrend),
            )


class Will_Rci(Bot):

    inlong = False
    inshort = False

    decimal_num = int(os.environ.get("BOT_DECIMAL_NUM", 3))
    price_decimal_num = int(os.environ.get("BOT_PRICE_DECIMAL_NUM", 2))
    lot_percent = 100 / int(os.environ.get("BOT_LOT_PERCENT", 10))
    take_profit_percent = 100 / int(os.environ.get("BOT_TAKE_PROFIT_PERCENT", 50))

    def __init__(self):
        Bot.__init__(self, "1m")

    def ohlcv_len(self):
        return 6790

    def options(self):
        return {
            "rcv_short_len": hp.quniform("rcv_short_len", 1, 21, 1),
            "rcv_medium_len": hp.quniform("rcv_medium_len", 21, 34, 1),
            "rcv_long_len": hp.quniform("rcv_long_len", 34, 55, 1),
        }

    def strategy(self, open, close, high, low, volume):
        # logger.info('strategy start ctime : %s' % time.ctime())
        # start = time.time()  # 시작 시간 저장
        lot = self.exchange.get_lot(asset="BUSD")
        lot = round(lot / self.lot_percent, self.decimal_num)

        pos_size = self.exchange.get_position_size()
        pos_margin = (abs(pos_size) * self.exchange.get_position_entry_price()) / self.exchange.get_leverage()
        tp_order = self.exchange.get_open_order("TP")

        itv_s = self.input("rcv_short_len", int, 21)
        itv_m = self.input("rcv_medium_len", int, 34)
        itv_l = self.input("rcv_long_len", int, 55)

        rci_s = rci(close, itv_s)
        rci_m = rci(close, itv_m)
        rci_l = rci(close, itv_l)

        ra = rci_s[-1] / 2 - 50
        rb = rci_m[-1] / 2 - 50
        rc = rci_l[-1] / 2 - 50

        # willr for five willilams
        a = willr(high, low, close, period=55)
        b = willr(high, low, close, period=144)
        c = willr(high, low, close, period=610)
        x = willr(high, low, close, period=4181)
        y = willr(high, low, close, period=6785)

        buycon1 = True if (a[-1] < -97 and (b[-1] < -97 or c[-1] < -97) and (x[-1] < -80 or y[-1] < -80)) else False
        buycon2 = True if (a[-1] < -97 and (b[-1] < -97 and c[-1] < -90) and (x[-1] > -35 or y[-1] > -35)) else False
        buycon3 = True if (a[-1] < -97 and (b[-1] < -97 and c[-1] > -70) and (x[-1] > -50 or y[-1] > -25)) else False
        buycon4 = True if (a[-1] < -97 and (b[-1] < -97 and c[-1] < -97) and (x[-1] > -50 or y[-1] > -50)) else False
        buycon5 = True if (a[-1] < -97 and (b[-1] < -97 and c[-1] < -75) and (x[-1] > -25 or y[-1] > -25)) else False
        buycon6 = True if ((b[-1] + 100) * (c[-1] + 100) == 0 and (c[-1] < -75 and x[-1] > -30 or y[-1] > -30)) else False
        buycon7 = True if ((b[-1] + 100) == 0 and (c[-1] > -30 and x[-1] > -30 or y[-1] > -30)) else False
        buycon8 = True if c[-1] < -97 else False
        buycon9 = True if a[-1] < -97 and b[-1] < -97 and c[-1] > -50 else False

        sellcon1 = True if (a[-1] > -3 and (b[-1] > -3 or c[-1] > -3) and (x[-1] > -20 or y[-1] > -20)) else False
        sellcon2 = True if (a[-1] > -3 and (b[-1] > -3 and c[-1] > -10) and (x[-1] < -65 or y[-1] < -65)) else False
        sellcon3 = True if (a[-1] > -3 and (b[-1] > -3 and c[-1] < -30) and (x[-1] < -50 or y[-1] < -75)) else False
        sellcon4 = True if (a[-1] > -3 and (b[-1] > -3 and c[-1] > -3) and (x[-1] < -50 or y[-1] < -50)) else False
        sellcon5 = True if (a[-1] > -3 and (b[-1] > -3 and c[-1] < -25) and (x[-1] < -75 or y[-1] < -75)) else False
        sellcon6 = True if (((b[-1]) * (c[-1])) == 0 and c[-1] > -25 and (x[-1] < -70 or y[-1] < -70)) else False
        sellcon7 = True if ((b[-1]) == 0 and (c[-1] < -70 and x[-1] < -70 or y[-1] < -70)) else False
        sellcon8 = True if c[-1] > -3 else False
        sellcon9 = True if a[-1] > -3 and b[-1] > -3 and c[-1] < -50 else False

        buyRCIfillerCon = True if rc < -80 else False
        sellRCIfillerCon = True if rc > -20 else False

        buyWillfilterCon = buycon1 or buycon2 or buycon3 or buycon4 or buycon5 or buycon6 or buycon7 or buycon8 or buycon9
        sellWillFilrerCon = sellcon1 or sellcon2 or sellcon3 or sellcon4 or sellcon5 or sellcon6 or sellcon7 or sellcon8 or sellcon9

        # set condition
        buyCons = buyWillfilterCon and buyRCIfillerCon
        sellCons = sellWillFilrerCon and sellRCIfillerCon

        buyCon = True if buyCons else False
        sellCon = True if sellCons else False

        # buyCloseCon = sellRCIfillerCon
        buyCloseCon = sellWillFilrerCon

        # sellCloseCon = buyRCIfillerCon
        sellCloseCon = buyWillfilterCon

        if not eval(os.environ.get("BOT_TEST", "False")):
            # self.exchange.exit(profit=(float(pos_margin / self.take_profit_percent)))
            if tp_order is None and pos_size != 0:
                if pos_size < 0:
                    # self.exchange.order("TP", True, abs(pos_size), take_profit=round(self.exchange.get_position_entry_price() * (1 - (1 / self.take_profit_percent) / self.exchange.get_leverage()), self.price_decimal_num), reduce_only=True)
                    self.exchange.order("TP", True, abs(pos_size), trailing_stop=1)
                if pos_size > 0:
                    # self.exchange.order("TP", False, abs(pos_size), take_profit=round(self.exchange.get_position_entry_price() * ((1 / self.take_profit_percent) / self.exchange.get_leverage() + 1), self.price_decimal_num), reduce_only=True)
                    self.exchange.order("TP", False, abs(pos_size), trailing_stop=1)

            # if (buyCloseCon and pos_size > 0) or (sellCloseCon and pos_size < 0):
            #     self.exchange.close_all()
            #     self.exchange.cancel_all()

            if buyCon and pos_size <= 0:
                self.exchange.order("Long", True, lot)
            if sellCon and pos_size >= 0:
                self.exchange.order("Short", False, lot)

        logger.info(f"--------------------------------------")

        logger.info(f"a:   {round(a[-1], 2)}")
        logger.info(f"b:   {round(b[-1], 2)}")
        logger.info(f"c:   {round(c[-1], 2)}")
        logger.info(f"x:   {round(x[-1], 2)}")
        logger.info(f"y:   {round(y[-1], 2)}")
        logger.info(f"rc:  {round(rc, 2)}")
        logger.info(f"lot: {round(lot, self.decimal_num)}")

        logger.info(f"--------------------------------------")

        logger.info(f"WILLR Buy conditions: {sum([buycon1, buycon2, buycon3, buycon4, buycon5, buycon6, buycon7, buycon8, buycon9])}/9")
        logger.info(f"WILLR Sell conditions: {sum([sellcon1, sellcon2, sellcon3, sellcon4, sellcon5, sellcon6, sellcon7, sellcon8, sellcon9])}/9")

        logger.info(f"RCI Buy conditions: {buyRCIfillerCon}")
        logger.info(f"RCI Sell conditions: {sellRCIfillerCon}")
        logger.info(f"In {'LONG' if pos_size > 0 else ('SHORT' if pos_size < 0 else 'no')} position")
        # if pos_size != 0:
        #     logger.info(f'{float(self.exchange.get_position()["unRealizedProfit"])} / {(pos_margin / self.take_profit_percent)}')

        # logger.info('all strategy processing time : %s' % str(time.time() - start))
