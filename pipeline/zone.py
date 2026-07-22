"""매수 구간(zone) 산출 (명세서 §2.5 ✅)."""
from __future__ import annotations

from . import config


def buy_zone(signal: str, ref_price: float) -> tuple[float, float]:
    """signal: "이평" | "박스" | "신고가". 반환: (zone_low, zone_high)."""
    if signal == "이평":
        return ref_price * config.ZONE_MA_LOW, ref_price * config.ZONE_MA_HIGH
    if signal in ("박스", "신고가"):
        return ref_price, ref_price * config.ZONE_BREAKOUT_HIGH
    raise ValueError(f"unknown signal: {signal}")


def in_zone(signal: str, ref_price: float, cur_price: float) -> bool:
    low, high = buy_zone(signal, ref_price)
    return low <= cur_price <= high
