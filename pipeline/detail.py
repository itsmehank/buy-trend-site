"""카드 상세(by-ticker) 응답 3종 생성 (명세서 §2.7 ✅)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest, config, indicators, signals


def _full_hold_row(closes: np.ndarray, event_idx: np.ndarray, hold: int) -> dict:
    """nhigh용 확장 성적 행 (avg_ret/median_ret 포함)."""
    event_idx = np.asarray(event_idx, dtype=int)
    valid = event_idx[event_idx + hold < len(closes)]
    if len(valid) == 0:
        return {"hold": hold, "n": 0, "win_rate": None, "avg_ret": None,
                "median_ret": None, "avg_win": None, "avg_loss": None, "pl_ratio": None}
    rets = closes[valid + hold] / closes[valid] - 1.0
    base = backtest.hold_stats(closes, event_idx, hold)
    return {"hold": hold, "n": base["n"], "win_rate": base["win_rate"],
            "avg_ret": round(float(rets.mean() * 100), 2),
            "median_ret": round(float(np.median(rets) * 100), 2),
            "avg_win": base["avg_win"], "avg_loss": base["avg_loss"],
            "pl_ratio": base["pl_ratio"]}


def ma_by_ticker(ticker: str, df: pd.DataFrame, benchmark: str, market_s: bool) -> dict:
    closes = df["close"].reset_index(drop=True)
    c_np = closes.to_numpy()
    long_ok, short_ok = signals.trend_context(closes)
    scan = signals.ma_signal_scan(closes)
    mas = []
    for ma_type, periods in (("SMA", config.SMA_PERIODS), ("EMA", config.EMA_PERIODS)):
        for p in periods:
            v = scan[f"L|{ma_type}|{p}"]["ma_value"]
            mas.append({
                "ma_type": ma_type, "ma_period": p,
                "ma_value": round(v, 4) if v else None,
                "distance_pct": round((c_np[-1] / v - 1.0) * 100, 2) if v else None,
            })
    stats = {}
    for key, s in scan.items():
        rows = backtest.backtest_signal(c_np, s["events"], config.HOLDS_MA)
        stats[key] = [{"period": r["hold"], "win_rate": r["win_rate"],
                       "avg_win": r["avg_win"], "avg_loss": r["avg_loss"],
                       "pl_ratio": r["pl_ratio"], "touch_count": r["n"]} for r in rows]
    return {
        "ticker": ticker, "found": True,
        "last_close": round(float(c_np[-1]), 4),
        "last_date": str(df["date"].iloc[-1]),
        "benchmark": benchmark, "market_s": market_s,
        "ticker_long": bool(long_ok.iloc[-1]), "ticker_short": bool(short_ok.iloc[-1]),
        "mas": mas, "stats": stats,
    }


def box_by_ticker(ticker: str, df: pd.DataFrame) -> dict:
    closes = df["close"].reset_index(drop=True)
    highs = df["high"].reset_index(drop=True)
    lows = df["low"].reset_index(drop=True)
    vols = df["volume"].reset_index(drop=True)
    c_np = closes.to_numpy()
    avg_vol20 = vols.rolling(20).mean().to_numpy()
    v_np = vols.to_numpy()

    def period_rows(events: np.ndarray, filt: str) -> list[dict]:
        rows = backtest.backtest_signal(c_np, events, config.HOLDS_BOX)
        return [{"filter": filt, "period": r["hold"], "win_rate": r["win_rate"],
                 "avg_win": r["avg_win"], "avg_loss": r["avg_loss"],
                 "pl_ratio": r["pl_ratio"], "cnt": r["n"]} for r in rows]

    periods_all, periods_vol2x = {}, {}
    for filt, min_days in (("L", config.BOX_MIN_DAYS_L), ("S", config.BOX_MIN_DAYS_S)):
        events, _ = signals.box_breakouts(highs, lows, closes, min_days)
        periods_all[filt] = period_rows(events, filt)
        vol2x = np.array([i for i in events
                          if not np.isnan(avg_vol20[i]) and v_np[i] >= 2 * avg_vol20[i]],
                         dtype=int)
        periods_vol2x[filt] = period_rows(vol2x, filt)
    return {
        "ticker": ticker, "found": True,
        "vol_data_available": bool(np.nansum(v_np) > 0),
        "periods_all": periods_all,
        "periods_vol2x": periods_vol2x,
        "breakout_vol_mult": 2,
    }


def nhigh_by_ticker(ticker: str, df: pd.DataFrame, market: str) -> dict:
    closes = df["close"].reset_index(drop=True)
    c_np = closes.to_numpy()
    scan = signals.nhigh_signal_scan(closes)
    entries = {}
    for w in config.NHIGH_WINDOWS:
        key = str(w)
        s = scan[key]
        prev = closes.shift(1)
        line_now = (prev.expanding().max() if w == "ATH"
                    else prev.rolling(w).max()).iloc[-1]
        distance_pct = (round(max(0.0, (float(line_now) / c_np[-1] - 1.0) * 100), 2)
                        if pd.notna(line_now) else None)
        entries[key] = {
            "distance_pct": distance_pct,
            "label": "매수가능" if s["current"] else "대기",
            "breakout_pct": s["breakout_pct"],
            "near_pct": None,
            "days_since_breakout": s["days_since"],
            "holds": [_full_hold_row(c_np, s["events"], h) for h in config.HOLDS_NHIGH],
        }
    current = [k for k, s in scan.items() if s["current"]]
    return {
        "ticker": ticker, "found": True, "market": market,
        "last_close": round(float(c_np[-1]), 4),
        "last_date": str(df["date"].iloc[-1]),
        "holds": config.HOLDS_NHIGH,
        "entries": entries,
        "signal": current[0] if current else None,
    }
