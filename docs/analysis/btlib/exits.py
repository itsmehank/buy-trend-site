"""매도 규칙 — 2026-07-29 검증의 수정본(US 스크립트) 의미론을 그대로 따른다.

모든 함수는 '진입 후 보유일수'를 돌려준다. 발동하지 않으면 maxh.
E 규칙(연속 하회)은 이벤트 이전에 시작된 연속을 이어받으면 안 된다(§5-① 버그).
"""
from __future__ import annotations

import numpy as np


def trailing_days(seg: np.ndarray, k: float, maxh: int) -> int:
    """트레일링 스톱: 보유 중 고점(진입일 포함) 대비 종가가 k배 이하로 처음 내려온 날."""
    rmax = np.maximum.accumulate(seg)
    w = np.flatnonzero(seg[1:] <= rmax[1:] * k)
    return int(w[0]) + 1 if len(w) else maxh


def below_sma_days(below_idx: np.ndarray, i: int, maxh: int) -> int:
    """이평 하회 청산: 전체 시계열에서 '종가 < 이평'인 인덱스 목록(below_idx, 오름차순)을
    받아, 진입 i 다음 날 이후 첫 하회일까지의 보유일수."""
    p = np.searchsorted(below_idx, i + 1, "left")
    return min(int(below_idx[p]) - i, maxh) if p < len(below_idx) else maxh


def run_lengths(below: np.ndarray) -> np.ndarray:
    """각 시점에서 끝나는 '연속 하회 일수'. 첫 원소는 0으로 센다(전일 없음)."""
    runlen = np.zeros(len(below), dtype=int)
    for k in range(1, len(below)):
        runlen[k] = runlen[k - 1] + 1 if below[k] else 0
    return runlen


def consec_below_days(runlen: np.ndarray, i: int, n_days: int, maxh: int) -> int:
    """n일 연속 하회 청산. 전역 runlen을 '이벤트 이후 경과일수'로 상한 처리해
    이벤트 전에 시작된 연속을 이어받지 않는다."""
    seg = runlen[i + 1:i + maxh + 1]
    seg_run = np.minimum(seg, np.arange(1, len(seg) + 1))
    w = np.flatnonzero(seg_run >= n_days)
    return int(w[0]) + 1 if len(w) else maxh
