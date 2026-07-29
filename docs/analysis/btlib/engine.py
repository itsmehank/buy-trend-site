"""이벤트 백테스트 엔진 — 2026-07-29 검증의 수정본 US 스크립트(us_phase2)와 동일 의미론.

거래 1건 = 진입 이벤트 1개 × 매도 규칙 1개. 재진입·포지션 중복 제어 없음(단일 거래 평가).
초과수익 = (종목 수익) − (같은 창의 벤치마크 수익), 비용 미차감(비용은 metrics에서).

필터 순서 (이벤트별):
  ① 이벤트 후 maxh봉 이력 존재  ② 진입일 벤치마크 존재
  ③ (옵션) PIT 유동성 ≥ min_dollar_vol  ④ (옵션) regime_map[진입일] == True
"""
from __future__ import annotations

import numpy as np

from . import entries as entries_mod
from . import exits, liquidity


def run_event_backtest(px, bench, entry_specs: dict, exit_specs: dict, *,
                       min_bars: int = 900, maxh: int = 504,
                       regime_map: dict | None = None,
                       min_dollar_vol: float | None = None,
                       dv_window: int = 20) -> dict:
    """px: long 포맷(ticker/date/OHLCV). bench: date/close DataFrame.

    entry_specs: {이름: entries.resolve_entry가 받는 spec}
    exit_specs:  {이름: ("fixed", h) | ("trailing", k) | ("sma_below", n)
                        | ("consec_below", (n, days))}
    반환: {진입: {매도: {"exc": [...], "days": [...]}}} + 특수 키 "_n_tickers"(처리 종목수)
    """
    bmap = dict(zip(bench["date"], bench["close"].astype(float)))
    entry_fns = {name: entries_mod.resolve_entry(s) for name, s in entry_specs.items()}
    trades = {e: {x: {"exc": [], "days": []} for x in exit_specs} for e in entry_specs}
    n_tickers = 0

    for t, g in px.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < min_bars:
            continue
        c = g["close"].to_numpy(float)
        if not np.all(c > 0):          # 종가 0/음수 종목(CBIO/DEC 유형) 제외
            continue
        n_tickers += 1
        dates = g["date"].to_numpy()
        bp = np.array([bmap.get(d, np.nan) for d in dates])
        cs = g["close"].reset_index(drop=True)
        dv = (liquidity.pit_dollar_vol(cs, g["volume"], dv_window)
              if min_dollar_vol is not None else None)

        # 청산 규칙이 요구하는 SMA 하회 인덱스·연속 카운트는 종목당 1회만 계산
        below_idx_cache, runlen_cache = {}, {}

        def sma_below(n):
            if n not in below_idx_cache:
                s = cs.rolling(n).mean().to_numpy()
                below = np.where(np.isnan(s), False, c < s)
                below_idx_cache[n] = np.flatnonzero(below)
                runlen_cache[n] = exits.run_lengths(below)
            return below_idx_cache[n], runlen_cache[n]

        for ename, fn in entry_fns.items():
            evs = np.asarray(fn(g), dtype=int)
            evs = evs[evs + maxh < len(c)]
            for i in evs:
                if np.isnan(bp[i]):
                    continue
                if dv is not None and not (dv[i] >= min_dollar_vol):
                    continue
                if regime_map is not None and not regime_map.get(dates[i], False):
                    continue
                seg = c[i:i + maxh + 1]
                for xname, (kind, param) in exit_specs.items():
                    if kind == "fixed":
                        h = int(param)
                        if h > maxh:
                            continue
                    elif kind == "trailing":
                        h = exits.trailing_days(seg, float(param), maxh)
                    elif kind == "sma_below":
                        bidx, _ = sma_below(int(param))
                        h = exits.below_sma_days(bidx, i, maxh)
                    elif kind == "consec_below":
                        n, days = param
                        _, rl = sma_below(int(n))
                        h = exits.consec_below_days(rl, i, int(days), maxh)
                    else:
                        raise ValueError(f"unknown exit rule: {kind}")
                    if np.isnan(bp[i + h]):
                        continue
                    r = trades[ename][xname]
                    r["exc"].append((c[i + h] / c[i] - 1.0) - (bp[i + h] / bp[i] - 1.0))
                    r["days"].append(h)

    trades["_n_tickers"] = n_tickers
    return trades
