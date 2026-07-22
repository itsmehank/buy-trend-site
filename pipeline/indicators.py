"""이동평균·롤링 지표 유틸."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(closes: pd.Series, period: int) -> pd.Series:
    return closes.rolling(period).mean()


def ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False, min_periods=period).mean()


def ma_series(closes: pd.Series, ma_type: str, period: int) -> pd.Series:
    return sma(closes, period) if ma_type == "SMA" else ema(closes, period)


def rolling_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).max()


def rolling_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).min()
