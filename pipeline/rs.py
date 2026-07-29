"""[자체 기준] 1: IBD식 RS — 0.4×3m + 0.2×6m + 0.2×9m + 0.2×12m → 시장별 백분위 0~100."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rs_raw(closes: np.ndarray) -> float | None:
    """RS 원점수. 12개월(252거래일) 이력이 없거나 값이 비유한이면 None.

    비유한값 방어: 룩백 시점 종가가 0이면 inf, 종가가 NaN이면 NaN이 된다.
    - NaN은 rs_percentiles의 int(round(...))에서 ValueError로 배치를 죽인다.
    - inf는 죽지 않고 rank 최상위가 되어 그 종목이 RS=100으로 상위 10%에
      무조건 진입한다(더 조용하고 더 위험하다).
    둘 다 '유니버스에서 제외'가 옳으므로 이력 부족과 같이 None으로 돌린다.
    """
    need = max(config.RS_WEIGHTS) + 1
    if len(closes) < need:
        return None
    raw = 0.0
    for lb, w in config.RS_WEIGHTS.items():
        raw += w * (closes[-1] / closes[-1 - lb] - 1.0)
    return float(raw) if np.isfinite(raw) else None


def rs_percentiles(raw_scores: dict[str, float]) -> dict[str, int]:
    """{ticker: raw} → {ticker: 0~100 백분위}. 동일 시장 유니버스 내에서 호출할 것."""
    if not raw_scores:
        return {}
    s = pd.Series(raw_scores)
    pct = s.rank(pct=True) * 100.0
    return {t: int(round(v)) for t, v in pct.items()}
