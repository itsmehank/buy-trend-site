"""백테스트 성적표 계산, EV 공식, 최적 보유기간 선정 (명세서 §2.4)."""
from __future__ import annotations

import numpy as np

from . import config


def hold_stats(closes: np.ndarray, event_idx: np.ndarray, hold: int) -> dict | None:
    """이벤트 발생일 종가 매수 → hold 거래일 뒤 종가 매도의 성적.

    반환: {win_rate, avg_win, avg_loss, pl_ratio, n} (%, avg_loss는 음수).
    미래 데이터가 없는 이벤트는 표본에서 제외. 손실 표본이 없으면
    avg_loss/pl_ratio = None (성적표에서 "—" 표시, EV 후보 제외).
    """
    event_idx = np.asarray(event_idx, dtype=int)
    valid = event_idx[event_idx + hold < len(closes)]
    if len(valid) == 0:
        return None
    rets = closes[valid + hold] / closes[valid] - 1.0
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    n = len(rets)
    win_rate = len(wins) / n * 100.0
    avg_win = float(wins.mean() * 100.0) if len(wins) else 0.0
    if len(losses) == 0 or losses.mean() == 0.0:
        avg_loss = None
        pl_ratio = None
    else:
        avg_loss = float(losses.mean() * 100.0)
        pl_ratio = round(avg_win / abs(avg_loss), 4)
    return {
        "hold": hold,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "pl_ratio": pl_ratio,
        "n": n,
    }


def backtest_signal(closes: np.ndarray, event_idx, holds: list[int]) -> list[dict]:
    """8개 보유기간 전체의 성적표. 표본 0인 기간은 n=0 행으로 유지."""
    rows = []
    for h in holds:
        s = hold_stats(closes, event_idx, h)
        if s is None:
            s = {"hold": h, "win_rate": None, "avg_win": None,
                 "avg_loss": None, "pl_ratio": None, "n": 0}
        rows.append(s)
    return rows


def ev(win_rate: float, avg_win: float, avg_loss: float | None) -> float | None:
    """EV = (p×avg_win + (1−p)×avg_loss) / |avg_loss|. 무손실이면 정의 불가(None)."""
    if avg_loss is None or avg_loss == 0:
        return None
    p = win_rate / 100.0
    return (p * avg_win + (1.0 - p) * avg_loss) / abs(avg_loss)


def select_optimal(period_rows: list[dict], min_sample: int = config.MIN_SAMPLE) -> dict | None:
    """표본 ≥ min_sample & 손실 표본 존재(avg_loss 있음)인 기간 중 EV 최대 기간 선택.

    반환: 선택된 기간 행 + {"ev": ...}. 후보 없으면 None.
    """
    best = None
    best_ev = None
    for row in period_rows:
        if row["n"] < min_sample or row["win_rate"] is None:
            continue
        e = ev(row["win_rate"], row["avg_win"], row["avg_loss"])
        if e is None:
            continue
        if best_ev is None or e > best_ev:
            best_ev = e
            best = row
    if best is None:
        return None
    return {**best, "ev": round(best_ev, 4)}
