import pandas as pd
from pandas import DataFrame, Series
from unicodedata import decimal


####################################################################
######################### volabitlity indicators ###################
####################################################################

def donchian(df: DataFrame, upper_period: int = 20, lower_period: int = 5
    ) -> DataFrame:

        upper = pd.Series(
            df["high"].rolling(center=False, window=upper_period).max(), name="donchian_uper"
        )
        lower = pd.Series(
            df["low"].rolling(center=False, window=lower_period).min(), name="donchian_lower"
        )
        middle = pd.Series((upper + lower) / 2, name="donchian_middle")

        return df.join(pd.concat([lower, middle, upper], axis=1), on="date")

def tr(df: DataFrame) -> Series:

        TR1 = pd.Series(df["high"] - df["low"]).abs()  # True Range = High less Low

        TR2 = pd.Series(
            df["high"] - df["close"].shift()
        ).abs()  # True Range = High less Previous Close

        TR3 = pd.Series(
            df["close"].shift() - df["low"]
        ).abs()  # True Range = Previous Close less Low

        _TR = pd.concat([TR1, TR2, TR3], axis=1)

        _TR["tr"] = _TR.max(axis=1)

        return df.join(pd.Series(_TR["tr"], name="tr").round(decimals=8), on="date")