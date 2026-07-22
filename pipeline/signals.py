"""신호 3종 검출 — 이평 눌림 / 박스 돌파 / 신고가 (명세서 §2.2 + [자체 기준] 2·3·4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, indicators


# ── 추세 맥락 (L/S) — [자체 기준] 2

def trend_context(closes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(long_ok, short_ok). L = 종가>SMA200 & SMA50>SMA200, S = 종가>SMA50."""
    sma50 = indicators.sma(closes, config.TREND_SHORT_SMA)
    sma200 = indicators.sma(closes, config.TREND_LONG_SMA)
    long_ok = (closes > sma200) & (sma50 > sma200)
    short_ok = closes > sma50
    return long_ok.fillna(False), short_ok.fillna(False)


# ── 이평 눌림

def ma_near(closes: pd.Series, ma: pd.Series) -> pd.Series:
    """종가가 이평선의 −2%~+3% 이내."""
    return ((closes >= ma * config.MA_NEAR_LOW)
            & (closes <= ma * config.MA_NEAR_HIGH)).fillna(False)


def touch_events(near: pd.Series, context: pd.Series) -> np.ndarray:
    """[자체 기준] 4: 근접 상태 '진입일'만 터치 1회. 연속 근접일은 재진입해야 새 터치."""
    entry = near & ~near.shift(1, fill_value=False)
    return np.flatnonzero((entry & context).to_numpy())


def ma_signal_scan(closes: pd.Series) -> dict:
    """24조합(SMA/EMA × 6기간 × L/S)의 이벤트 인덱스·현재 상태·이평값.

    반환: {"{L|S}|{SMA|EMA}|{p}": {"events": ndarray, "current": bool, "ma_value": float}}
    """
    long_ok, short_ok = trend_context(closes)
    out = {}
    for ma_type, periods in (("SMA", config.SMA_PERIODS), ("EMA", config.EMA_PERIODS)):
        for p in periods:
            ma = indicators.ma_series(closes, ma_type, p)
            near = ma_near(closes, ma)
            for filt, ctx in (("L", long_ok), ("S", short_ok)):
                key = f"{filt}|{ma_type}|{p}"
                out[key] = {
                    "events": touch_events(near, ctx),
                    "current": bool(near.iloc[-1] and ctx.iloc[-1]),
                    "ma_value": float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else None,
                }
    return out


# ── 박스 돌파

def box_breakouts(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                  min_days: int) -> tuple[np.ndarray, np.ndarray]:
    """박스(직전 min_days일 고저 진폭 ≤15%) 상단 종가 돌파.

    재돌파는 박스 재형성(진폭 조건 재충족) 후에만 인정.
    반환: (event_idx, box_top[event]) — box_top = 직전 min_days일 최고가.
    """
    top = highs.rolling(min_days).max().shift(1)
    bot = lows.rolling(min_days).min().shift(1)
    amp_ok = ((top - bot) / bot <= config.BOX_AMP_MAX).fillna(False).to_numpy()
    top_np = top.to_numpy()
    c = closes.to_numpy()
    cand = amp_ok & (c > top_np)
    events, tops = [], []
    armed = True  # 박스 형성 상태에서 첫 돌파만 인정
    for i in np.flatnonzero(cand | ~amp_ok):
        if not amp_ok[i]:
            armed = True  # 박스가 깨졌다 재형성되면 다시 무장
            continue
        if armed:
            events.append(i)
            tops.append(top_np[i])
            armed = False
    return np.asarray(events, dtype=int), np.asarray(tops, dtype=float)


def box_signal_scan(highs: pd.Series, lows: pd.Series, closes: pd.Series) -> dict:
    """L(60일)/S(20일) 박스 각각의 이벤트와 현재 신호 여부."""
    out = {}
    n = len(closes)
    for filt, min_days in (("L", config.BOX_MIN_DAYS_L), ("S", config.BOX_MIN_DAYS_S)):
        events, tops = box_breakouts(highs, lows, closes, min_days)
        current, ref = False, None
        if len(events):
            last, last_top = int(events[-1]), float(tops[-1])
            if n - 1 - last <= config.BREAKOUT_MAX_AGE and closes.iloc[-1] >= last_top:
                current, ref = True, last_top
        out[filt] = {"events": events, "tops": tops, "current": current,
                     "ref_price": ref,
                     "days_since": (n - 1 - int(events[-1])) if len(events) else None}
    return out


# ── 신고가

def nhigh_breakouts(closes: pd.Series, window) -> tuple[np.ndarray, np.ndarray]:
    """신고가 돌파(크로싱) 이벤트. window = 정수(20/55) 또는 "ATH".

    돌파 1회 = 직전 최고가를 밑에서 위로 넘어선 날. 연속 신고가 갱신은
    크로싱이 아니므로 자동으로 1회만 카운트된다.
    반환: (event_idx, 돌파선[event]) — 돌파선 = 직전 최고 종가.
    """
    prev = closes.shift(1)
    if window == "ATH":
        prior_max = prev.expanding().max()
    else:
        prior_max = prev.rolling(window).max()
    above = (closes > prior_max).fillna(False)
    cross = (above & ~above.shift(1, fill_value=False)).to_numpy()
    idx = np.flatnonzero(cross)
    return idx, prior_max.to_numpy()[idx]


def nhigh_signal_scan(closes: pd.Series) -> dict:
    """ATH/20/55 각각의 이벤트와 현재 신호 여부."""
    out = {}
    n = len(closes)
    for w in config.NHIGH_WINDOWS:
        events, lines = nhigh_breakouts(closes, w)
        current, ref, days_since, breakout_pct = False, None, None, None
        if len(events):
            last, line = int(events[-1]), float(lines[-1])
            days_since = n - 1 - last
            breakout_pct = round((closes.iloc[last] / line - 1.0) * 100.0, 2)
            if days_since <= config.BREAKOUT_MAX_AGE and closes.iloc[-1] >= line:
                current, ref = True, line
        out[str(w)] = {"events": events, "lines": lines, "current": current,
                       "ref_price": ref, "days_since": days_since,
                       "breakout_pct": breakout_pct}
    return out
