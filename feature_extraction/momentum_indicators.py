import pandas as pd
from pandas import DataFrame, Series
from sklearn.manifold import smacof

####################################################################
######################### momentum indicators ######################
####################################################################

#Trix (TRIX)
# https://www.tradingview.com/wiki/TRIX
def trix(df: DataFrame,
        period: int = 20,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:
        data = df[column]

        def _ema(data, period, adjust):
            return pd.Series(data.ewm(span=period, adjust=adjust).mean())

        m = _ema(_ema(_ema(data, period, adjust), period, adjust), period, adjust)

        trix_serise = pd.Series(100 * (m.diff() / m), name="{0} period trix".format(period))

        return df.join(trix_serise, on="date")

#Williams %R, or just %R
def willsr(df: DataFrame, period: int = 14) -> Series:

        highest_high = df["high"].rolling(center=False, window=period).max()
        lowest_low = df["low"].rolling(center=False, window=period).min()

        WR = pd.Series(
            (highest_high - df["close"]) / (highest_high - lowest_low),
            name="{0} williams %R".format(period),
        )

        wr_series = WR * -100
        return df.join(wr_series.round(decimals=8), on="date")

# The Awesome Oscillator(AO)
# https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
def ao(df: DataFrame, slow_period: int = 34, fast_period: int = 5) -> Series:
        slow = pd.Series(
            ((df["high"] + df["low"]) / 2).rolling(window=slow_period).mean(),
            name="slow_AO",
        )
        fast = pd.Series(
            ((df["high"] + df["low"]) / 2).rolling(window=fast_period).mean(),
            name="fast_AO",
        )
        ao_series =  pd.Series(fast - slow, name="ao")
        return df.join(ao_series.round(decimals=8), on="date")

#True Strength Index (TSI)
def tsi(df: DataFrame,
        long: int = 25,
        short: int = 13,
        signal: int = 13,
        column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:

        ## Double smoother price change
        momentum = pd.Series(df[column].diff())  ## 1 period momentum
        _EMA25 = pd.Series(
            momentum.ewm(span=long, min_periods=long - 1, adjust=adjust).mean(),
            name="_price change EMA25",
        )
        _DEMA13 = pd.Series(
            _EMA25.ewm(span=short, min_periods=short - 1, adjust=adjust).mean(),
            name="_price change double smoothed DEMA13",
        )

        ## Double smoothed absolute price change
        absmomentum = pd.Series(df[column].diff().abs())
        _aEMA25 = pd.Series(
            absmomentum.ewm(span=long, min_periods=long - 1, adjust=adjust).mean(),
            name="_abs_price_change EMA25",
        )
        _aDEMA13 = pd.Series(
            _aEMA25.ewm(span=short, min_periods=short - 1, adjust=adjust).mean(),
            name="_abs_price_change double smoothed DEMA13",
        )

        TSI = pd.Series((_DEMA13 / _aDEMA13) * 100, name="tsi")
        signal = pd.Series(
            TSI.ewm(span=signal, min_periods=signal - 1, adjust=adjust).mean(),
            name="tsi_signal",
        )

        return df.join(pd.concat([TSI, signal], axis=1).round(decimals=8), on="date")

#Chande Momentum Oscillator (CMO)
#https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/
def cmo(df: DataFrame,
        period: int = 9,
        factor: int = 100,
        column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:
        # get the price diff
        delta = df[column].diff()

        # positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # EMAs of ups and downs
        _gain = up.ewm(com=period, adjust=adjust).mean()
        _loss = down.ewm(com=period, adjust=adjust).mean().abs()

        cmo_serise = pd.Series(factor * ((_gain - _loss) / (_gain + _loss)), name="cmo")
        return df.join(cmo_serise, on="date")


#The Stochastic Oscillator (STOCH)
#https://www.tradingview.com/wiki/Stochastic_(STOCH)
def stoch(df: DataFrame, period: int = 14) -> Series:

        highest_high = df["high"].rolling(center=False, window=period).max()
        lowest_low = df["low"].rolling(center=False, window=period).min()

        STOCH = pd.Series(
            (df["close"] - lowest_low) / (highest_high - lowest_low) * 100,
            name="{0} period stoch %K".format(period),
        )

        return df.join(STOCH.round(decimals=8), on="date")
        
# Fisher Transform (FISHT)
def fisher(df: DataFrame, period: int = 10, adjust: bool = True) -> Series:

        from numpy import log, seterr

        seterr(divide="ignore")

        med = (df["high"] + df["low"]) / 2
        ndaylow = med.rolling(window=period).min()
        ndayhigh = med.rolling(window=period).max()
        raw = (2 * ((med - ndaylow) / (ndayhigh - ndaylow))) - 1
        smooth = raw.ewm(span=5, adjust=adjust).mean()
        _smooth = smooth.fillna(0)

        figher_serise =  pd.Series(
            (log((1 + _smooth) / (1 - _smooth))).ewm(span=3, adjust=adjust).mean(),
            "fisht",
        )
        return df.join(figher_serise.round(decimals=8), on="date")

# The Moving Average Convergence Divergence (MACD)
# https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
def macd(df: DataFrame,
        period_fast: int = 12,
        period_slow: int = 26,
        signal: int = 9,
        column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:
        """
       
        """

        EMA_fast = pd.Series(
            df[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            df[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="macd")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="macd_signal"
        )

        the_serise =  pd.concat([MACD, MACD_signal], axis=1)
        return df.join(the_serise.round(decimals=8), on="date")

#Momentum (MOM) is an indicator used to measure a security's speed (or strength) of movement.
def mom(df: DataFrame, period: int = 10, column: str = "close") -> Series:
        return df.join(pd.Series(df[column].diff(period), name="mom").round(decimals=8), on="date")

# Percentage Price Oscillator (PPO)
# https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
def ppo(df: DataFrame,
        period_fast: int = 12,
        period_slow: int = 26,
        signal: int = 9,
        column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:

        EMA_fast = pd.Series(
            df[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            df[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        PPO = pd.Series(((EMA_fast - EMA_slow) / EMA_slow) * 100, name="ppo")
        PPO_signal = pd.Series(
            PPO.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="ppo_signal"
        )
        PPO_histo = pd.Series(PPO - PPO_signal, name="ppo_histo")

        return df.join(pd.concat([PPO, PPO_signal, PPO_histo], axis=1).round(decimals=8), on="date") 

#Rate of Change (ROC)
#https://www.tradingview.com/wiki/Rate_of_Change_(ROC)
def roc(df: DataFrame, period: int = 12, column: str = "close") -> Series:
        roc_serise = pd.Series(
            (df[column].diff(period) / df[column].shift(period)) * 100, name="roc"
        )
        return df.join(roc_serise.round(decimals=8), on="date")

#The Relative Strength Index(RSI)
#https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)
def rsi(df: DataFrame,
        period: int = 14,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:
        ## get the price diff
        delta = df[column].diff()

        ## positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # EMAs of ups and downs
        _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()

        RS = _gain / _loss
        return df.join(pd.Series(100 - (100 / (1 + RS)), name="rsi").round(decimals=8), on="date")

#The Schaff Trend Cycle (Oscillator) 
def stc(df: DataFrame,
        period_fast: int = 23,
        period_slow: int = 50,
        k_period: int = 10,
        d_period: int = 3,
        column: str = "close",
        adjust: bool = True
    ) -> Series:
        EMA_fast = pd.Series(
            df[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )

        EMA_slow = pd.Series(
            df[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )

        MACD = pd.Series((EMA_fast - EMA_slow), name="MACD")

        STOK = pd.Series((
            (MACD - MACD.rolling(window=k_period).min())
            / (MACD.rolling(window=k_period).max() - MACD.rolling(window=k_period).min())
            ) * 100)

        STOD = STOK.rolling(window=d_period).mean()
        STOD_DoubleSmooth = STOD.rolling(window=d_period).mean()  # "double smoothed"
        stc_serise = pd.Series(STOD_DoubleSmooth, name="stc")
        return df.join(stc_serise.round(decimals=2), on="date")



