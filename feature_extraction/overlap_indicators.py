import pandas as pd
import numpy as np
from pandas import DataFrame, Series

####################################################################
######################### overlap indicators #######################
####################################################################

#WMA stands for weighted moving average.
def wma(df: DataFrame, period: int = 9, column: str = "close") -> Series:

        d = (period * (period + 1)) / 2  # denominator
        weights = np.arange(1, period + 1)

        def linear(w):
            def _compute(x):
                return (w * x).sum() / d

            return _compute

        _close = df[column].rolling(period, min_periods=period)
        wma = _close.apply(linear(weights), raw=True)

        return df.join(pd.Series(wma, name="{0} period wma.".format(period)).round(decimals=8), on="date")

#Exponential Weighted Moving Average
def ema(df: DataFrame,
        period: int = 9,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:

        return df.join(pd.Series(
            df[column].ewm(span=period, adjust=adjust).mean(),
            name="{0} period ema".format(period),
        ), on="date")

def kama(df: DataFrame,
        er: int = 10,
        ema_fast: int = 2,
        ema_slow: int = 30,
        period: int = 20,
        column: str = "close",
    ) -> Series:
        er = er(df, er)
        fast_alpha = 2 / (ema_fast + 1)
        slow_alpha = 2 / (ema_slow + 1)
        sc = pd.Series(
            (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2,
            name="smoothing_constant",
        )  ## smoothing constant

        sma = pd.Series(
            df[column].rolling(period).mean(), name="SMA"
        )  ## first KAMA is SMA
        kama = []
        # Current KAMA = Prior KAMA + smoothing_constant * (Price - Prior KAMA)
        for s, ma, price in zip(
            sc.iteritems(), sma.shift().iteritems(), df[column].iteritems()
        ):
            try:
                kama.append(kama[-1] + s[1] * (price[1] - kama[-1]))
            except (IndexError, TypeError):
                if pd.notnull(ma[1]):
                    kama.append(ma[1] + s[1] * (price[1] - ma[1]))
                else:
                    kama.append(None)

        sma["kama"] = pd.Series(
            kama, index=sma.index, name="{0} period kama.".format(period)
        )  ## apply the kama list to existing index
        return df.join(sma["kama"].round(decimals=8), on="date")

def er(df: DataFrame, period: int = 10, column: str = "close") -> Series:

        change = df[column].diff(period).abs()
        volatility = df[column].diff().abs().rolling(window=period).sum()

        return df.join(pd.Series(change / volatility, name="{0} period er".format(period)).round(decimals=8), on="date")