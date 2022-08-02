import pandas as pd
from pandas import DataFrame, Series

####################################################################
######################### trend indicators #########################
####################################################################

def qstick(df: DataFrame, period: int = 14) -> Series:
        _close = df["close"].tail(period)
        _open = df["open"].tail(period)

        return df.join(pd.Series(
            (_close - _open) / period, name="{0} period qstick.".format(period)
        ).round(decimals=8), on="date")

def psar(df: DataFrame, iaf: int = 0.02, maxaf: int = 0.2) -> DataFrame:
        length = len(df)
        high, low, close = df.high, df.low, df.close
        psar = close[0 : len(close)]
        psarbull = [None] * length
        psarbear = [None] * length
        bull = True
        af = iaf
        hp = high[0]
        lp = low[0]

        for i in range(2, length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

            reverse = False

            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf

            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + iaf, maxaf)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + iaf, maxaf)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]

            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]

        psar = pd.Series(psar, name="psar", index=df.index)
        psarbear = pd.Series(psarbull, name="psarbear", index=df.index)
        psarbull = pd.Series(psarbear, name="psarbull", index=df.index)

        return df.join(pd.concat([psar, psarbull, psarbear], axis=1).round(decimals=8), on="date")
