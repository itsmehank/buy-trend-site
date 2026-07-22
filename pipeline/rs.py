"""[자체 기준] 1: IBD식 RS — 0.4×3m + 0.2×6m + 0.2×9m + 0.2×12m → 시장별 백분위 0~100."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rs_raw(closes: np.ndarray) -> float | None:
    """RS 원점수. 12개월(252거래일) 이력이 없으면 None."""
    need = max(config.RS_WEIGHTS) + 1
    if len(closes) < need:
        return None
    raw = 0.0
    for lb, w in config.RS_WEIGHTS.items():
        raw += w * (closes[-1] / closes[-1 - lb] - 1.0)
    return float(raw)


def rs_percentiles(raw_scores: dict[str, float]) -> dict[str, int]:
    """{ticker: raw} → {ticker: 0~100 백분위}. 동일 시장 유니버스 내에서 호출할 것."""
    if not raw_scores:
        return {}
    s = pd.Series(raw_scores)
    pct = s.rank(pct=True) * 100.0
    return {t: int(round(v)) for t, v in pct.items()}
