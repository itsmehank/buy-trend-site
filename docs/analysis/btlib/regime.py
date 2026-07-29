"""시장 레짐 — 기본 정의는 파이프라인과 동일: 벤치마크 종가 > SMA200 (config.REGIME_SMA)."""
from __future__ import annotations

import pandas as pd


def bull_map(bench: pd.DataFrame, sma: int = 200) -> dict:
    """{date: bool}. SMA 워밍업 구간(NaN)은 False(보수적 — 강세 불명이면 매수 안 함)."""
    b = bench.sort_values("date").reset_index(drop=True)
    ma = b["close"].rolling(sma).mean()
    bull = (b["close"] > ma).fillna(False)
    return {d: bool(v) for d, v in zip(b["date"], bull)}
