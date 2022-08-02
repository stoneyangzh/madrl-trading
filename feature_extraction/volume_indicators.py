import pandas as pd
from pandas import DataFrame, Series

####################################################################
######################### volume indicators ########################
####################################################################

def tp(df: DataFrame) -> Series:
    return df.join(pd.Series((df["high"] + df["low"] + df["close"]) / 3, name="tp").round(decimals=8), on="date")

def efi(df: DataFrame,
        period: int = 13,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:
        """Elder's Force Index"""

        # https://tradingsim.com/blog/elders-force-index/
        fi = pd.Series(df[column].diff() * df["volume"])
        return df.join(pd.Series(
            fi.ewm(ignore_na=False, span=period, adjust=adjust).mean(),
            name="{0} period Force Index".format(period),
        ), on="date")
