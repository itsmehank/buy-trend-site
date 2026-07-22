"""별점 로직 — 기본 ★★★에서 감점 태그로 강등 (명세서 §2.6 ✅ + [자체 기준] 5)."""
from __future__ import annotations

import numpy as np

from . import config

TAG_VOL_Q4 = "변동성 최상위(Q4)"
TAG_WICK = "윗꼬리 5일+"
TAG_RS_HOT = "RS최고 97+(초과열)"
TAG_SURGE = "대형주 무리한급등(시총대형 & 직전20일≥+50% | 60일최대일간≥+20%)"

MINOR_TAGS = {TAG_VOL_Q4, TAG_WICK, TAG_RS_HOT}


def volatility_60d(closes: np.ndarray) -> float | None:
    """60일 일간수익률 표준편차. 데이터 부족 시 None."""
    if len(closes) < config.VOL_WINDOW + 1:
        return None
    rets = np.diff(closes[-(config.VOL_WINDOW + 1):]) / closes[-(config.VOL_WINDOW + 1):-1]
    return float(np.std(rets, ddof=1))


def upper_wick_days(opens, highs, closes, lookback=config.WICK_LOOKBACK) -> int:
    """최근 lookback일 중 윗꼬리가 몸통보다 큰 날 수."""
    o = np.asarray(opens[-lookback:], dtype=float)
    h = np.asarray(highs[-lookback:], dtype=float)
    c = np.asarray(closes[-lookback:], dtype=float)
    body = np.abs(c - o)
    wick = h - np.maximum(o, c)
    return int(np.sum(wick > body))


def surge_flags(closes: np.ndarray) -> bool:
    """직전 20일 수익률 ≥ +50% 또는 60일 내 최대 일간수익률 ≥ +20%."""
    if len(closes) < 21:
        return False
    ret20 = closes[-1] / closes[-21] - 1.0
    if ret20 >= config.SURGE_20D:
        return True
    tail = closes[-(config.VOL_WINDOW + 1):]
    daily = np.diff(tail) / tail[:-1]
    return bool(len(daily) and daily.max() >= config.SURGE_1D_MAX)


def compute_tags(*, vol: float | None, vol_q4_cut: float | None, wick_days: int,
                 rs: int, market_cap_usd: float | None, surged: bool) -> list[str]:
    """감점 태그 목록 (순서 고정)."""
    tags = []
    if vol is not None and vol_q4_cut is not None and vol >= vol_q4_cut:
        tags.append(TAG_VOL_Q4)
    if wick_days >= config.WICK_MIN_DAYS:
        tags.append(TAG_WICK)
    if rs >= config.RS_OVERHEAT:
        tags.append(TAG_RS_HOT)
    if (market_cap_usd is not None and market_cap_usd >= config.LARGE_CAP_USD and surged):
        tags.append(TAG_SURGE)
    return tags


def stars_from_tags(tags: list[str], asset: str) -> tuple[int | None, str | None]:
    """(stars, cut_reason). ETF는 (None, None)이 아니라 stars=None, cut_reason은 태그대로."""
    cut_reason = " / ".join(tags) if tags else None
    if asset == "etf":
        return None, cut_reason
    if TAG_SURGE in tags:
        return 1, cut_reason
    if any(t in MINOR_TAGS for t in tags):
        return 2, cut_reason
    return 3, cut_reason
