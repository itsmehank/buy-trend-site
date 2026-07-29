"""진입 신호 — 파이프라인의 검증된 스캔 함수를 재사용하고, 새 가설용 제네릭 규칙을 추가.

spec 형식 (name, param):
  ("신고가", 20 | 55 | "ATH")        — 신고가 크로싱 돌파
  ("박스", 60 | 20)                  — 박스(진폭≤15%) 상단 종가 돌파
  ("이평눌림", "L|SMA|20" 등)        — 이평 근접 터치 (signals.ma_signal_scan 키)
  ("이평선 상향돌파", n)             — 종가가 SMA(n)을 밑에서 위로 크로싱
  callable(g) -> ndarray             — 커스텀 (가설 스크립트에서 정의)

반환: 이벤트 인덱스 ndarray (g는 날짜순 정렬된 단일 종목 DataFrame).
"""
from __future__ import annotations

import numpy as np

from pipeline import config, signals

#: 2026-07-29 검증에서 쓴 표준 6종 (registry H-002~007)
STANDARD_ENTRIES = {
    "이평 L|SMA|20": ("이평눌림", "L|SMA|20"),
    "박스 L(60일)": ("박스", config.BOX_MIN_DAYS_L),
    "박스 S(20일)": ("박스", config.BOX_MIN_DAYS_S),
    "신고가 ATH": ("신고가", "ATH"),
    "신고가 20일": ("신고가", 20),
    "신고가 55일": ("신고가", 55),
}


def _ma_touch(g, key: str) -> np.ndarray:
    """ma_signal_scan에서 key 하나만 계산 (동일 로직, 24조합 전체 계산 회피)."""
    filt, ma_type, p = key.split("|")
    cs = g["close"].reset_index(drop=True)
    long_ok, short_ok = signals.trend_context(cs)
    ctx = long_ok if filt == "L" else short_ok
    from pipeline import indicators
    ma = indicators.ma_series(cs, ma_type, int(p))
    near = signals.ma_near(cs, ma)
    return signals.touch_events(near, ctx)


def resolve_entry(spec):
    if callable(spec):
        return spec
    kind, param = spec
    if kind == "신고가":
        return lambda g: signals.nhigh_breakouts(g["close"].reset_index(drop=True), param)[0]
    if kind == "박스":
        return lambda g: signals.box_breakouts(
            g["high"].reset_index(drop=True), g["low"].reset_index(drop=True),
            g["close"].reset_index(drop=True), param)[0]
    if kind == "이평눌림":
        return lambda g: _ma_touch(g, param)
    if kind == "이평선 상향돌파":
        def f(g, n=int(param)):
            cs = g["close"].reset_index(drop=True)
            ma = cs.rolling(n).mean()
            above = ((cs > ma) & ma.notna()).fillna(False)
            cross = (above & ~above.shift(1, fill_value=False)).to_numpy()
            return np.flatnonzero(cross)
        return f
    raise ValueError(f"unknown entry spec: {spec}")
