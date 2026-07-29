"""진입 시점 기준(PIT) 유동성 — universe.dollar_vol_filter의 tail(20)은 '오늘' 기준이라
과거 이벤트에 적용하면 look-ahead가 된다(2026-07-29 §5-②). 여기서는 각 시점의
직전 window봉 평균 거래대금을 그 자리에서 계산한다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pit_dollar_vol(close: pd.Series, volume: pd.Series, window: int = 20) -> np.ndarray:
    close = pd.Series(np.asarray(close, dtype=float))
    volume = pd.Series(np.asarray(volume, dtype=float))
    return (close * volume).rolling(window, min_periods=window).mean().to_numpy()
