"""성과 요약 — 초과수익(벤치마크 차감) 리스트를 받아 평균/중위/승률/비용후 연율화를 계산.

연율화는 원본과 같은 선형 환산(mean% × 연간회전수 − 회전수 × 비용%)이다.
짧은 보유를 기계적으로 유리하게 만드는 한계가 있으므로(2026-07-29 §8-3)
결과 문서에 반드시 함께 명시할 것.
"""
from __future__ import annotations

import numpy as np


def summarize(exc, days, cost_pcts=(0.10, 0.28)) -> dict:
    """exc: 소수 초과수익 리스트(0.05 = 5%). days: 보유일수 리스트."""
    ex = np.asarray(exc, dtype=float) * 100.0
    dy = np.asarray(days, dtype=float)
    if len(ex) == 0:
        return {"n": 0}
    tpy = 252.0 / dy.mean()
    return {
        "n": int(len(ex)),
        "mean_pct": float(ex.mean()),
        "median_pct": float(np.median(ex)),
        "winrate_pct": float((ex > 0).mean() * 100.0),
        "avg_days": float(dy.mean()),
        "net_ann_pct": {f"{c:.2f}": float(ex.mean() * tpy - tpy * c) for c in cost_pcts},
    }
