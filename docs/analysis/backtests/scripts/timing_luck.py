"""H-016 — 리밸런싱 시점 분산(staggered tranche)의 효과를 **측정**한다.

이 가설은 검정 대상이 아니라 측정 대상이다.

  트랜치 포트폴리오 ≈ 서로 다른 리밸일 변형들의 **평균**이고,
  평균의 분산은 `Var(mean) = (1/m²)·ΣΣ Cov(i,j)` 로 **산술적으로 결정**된다.
  즉 "줄어드는가"는 물을 필요가 없다 — 반드시 줄어든다.
  물어야 할 것은 **"얼마나 줄어드는가"** 이고, 그 크기는 변형 간 상관이 정한다.

  변형들이 완전히 상관되면(ρ=1) 감소는 0이고, 무상관이면 분산이 1/m 로 준다.

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/timing_luck.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/timing_luck.py --measure
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from btlib import loading, regime

sys.path.insert(0, "docs/analysis/backtests/scripts")
import correlation_cap as cc                                    # noqa: E402

N_TRANCHE = 4          # 월 4개 변형 (매월 1~4번째 주)
TRADING_WK = 5         # 주당 거래일


def offset_reb_indices(dates: np.ndarray, start_i: int, offset: int) -> list[int]:
    """각 달에서 `offset`번째 주의 마지막 거래일 인덱스."""
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    out = []
    for _, grp in s.groupby([s.index.year, s.index.month]):
        idx = grp.values
        pos = min((offset + 1) * TRADING_WK - 1, len(idx) - 1)
        if idx[pos] >= start_i:
            out.append(int(idx[pos]))
    return out


def run_variant(P, bull, cfg, rebs: list[int]) -> pd.Series:
    """베이스라인 B(40위 버퍼)를 주어진 리밸일로 돌려 일별 수익률 계열을 낸다."""
    c = P["close_ff"]
    held: list[int] = []
    n_days = len(P["dates"])
    ret = np.zeros(n_days)
    reb_set = {r: i for i, r in enumerate(rebs)}
    cur: list[int] = []
    for t in range(rebs[0] + 1, n_days):
        if cur:
            with np.errstate(all="ignore"):
                r = c[t][cur] / c[t - 1][cur] - 1.0
            r = r[np.isfinite(r)]
            ret[t] = float(r.mean()) if len(r) else 0.0
        if (t - 1) in reb_set:                       # 판단일 종가 → 익일부터 반영
            a = t - 1
            if not bull[a]:
                held, cur = [], []
            else:
                held, _ = cc.rebalance(P, a, held, cfg, None)
                cur = list(held)
    return pd.Series(ret[rebs[0] + 1:], index=pd.to_datetime(P["dates"][rebs[0] + 1:]))


def measure(market: str) -> dict:
    P = cc.build_panel(market)
    cfg = cc.MARKET_CFG[market]
    bmap = regime.bull_map(loading.load_bench(market), sma=cc.SMA_BENCH)
    bull = np.array([bmap.get(d, False) for d in P["dates"]])
    start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= cc.MIN_BARS)),
                cc.SMA_BENCH, cc.LB_FAR, 1)
    series = []
    for off in range(N_TRANCHE):
        rebs = offset_reb_indices(P["dates"], start, off)
        series.append(run_variant(P, bull, cfg, rebs))
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    M = np.column_stack([s.reindex(idx).fillna(0.0).to_numpy() for s in series])
    tranche = M.mean(axis=1)                         # 트랜치 = 변형들의 평균

    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = [(np.prod(1 + M[:, j]) ** (1 / yrs) - 1) * 100 for j in range(N_TRANCHE)]
    vol = [M[:, j].std(ddof=1) * np.sqrt(252) * 100 for j in range(N_TRANCHE)]
    corr = np.corrcoef(M.T)
    off_diag = corr[~np.eye(N_TRANCHE, dtype=bool)]
    return {"cagr": cagr, "vol": vol, "rho": float(off_diag.mean()),
            "tranche_cagr": (np.prod(1 + tranche) ** (1 / yrs) - 1) * 100,
            "tranche_vol": tranche.std(ddof=1) * np.sqrt(252) * 100,
            "years": yrs, "n_days": len(idx)}


def var_reduction(rho: float, m: int = N_TRANCHE) -> float:
    """등상관 m개 변형을 평균했을 때의 분산 비 Var(mean)/Var(개별)."""
    return (1 + (m - 1) * rho) / m


def cmd_measure():
    print("=" * 92)
    print("[H-016 타이밍 럭 측정] 리밸일 변형 4개의 산포와 트랜치 평균의 효과")
    print("=" * 92)
    for mk in ("kr", "us"):
        r = measure(mk)
        sd_cagr = float(np.std(r["cagr"], ddof=1))
        print(f"\n── {mk.upper()} ──  {r['years']:.1f}년 · {r['n_days']}거래일")
        print(f"  변형별 CAGR   : " + " / ".join(f"{v:.2f}%" for v in r["cagr"]))
        print(f"  변형별 연변동성: " + " / ".join(f"{v:.2f}%" for v in r["vol"]))
        print(f"  **타이밍 럭(CAGR 표준편차) = {sd_cagr:.2f}%p** "
              f"(최대−최소 {max(r['cagr']) - min(r['cagr']):.2f}%p)")
        print(f"  변형 간 평균 상관 ρ = {r['rho']:.4f}")
        print(f"  트랜치(4개 평균): CAGR {r['tranche_cagr']:.2f}% · "
              f"연변동성 {r['tranche_vol']:.2f}%")
        vr = var_reduction(r["rho"])
        print(f"  이론 분산비 (1+3ρ)/4 = {vr:.4f} → 변동성 {(1-np.sqrt(vr))*100:.2f}% 감소")
        print(f"  실측 변동성 감소     = "
              f"{(1 - r['tranche_vol'] / np.mean(r['vol'])) * 100:.2f}%")


def selftest():
    # ① 분산비 극단값
    assert abs(var_reduction(1.0) - 1.0) < 1e-12          # 완전상관 → 감소 없음
    assert abs(var_reduction(0.0) - 0.25) < 1e-12         # 무상관 → 1/4
    assert var_reduction(0.5) == (1 + 3 * 0.5) / 4
    # ② 단조성 — ρ가 클수록 감소 폭이 작다
    vs = [var_reduction(r) for r in (0.0, 0.3, 0.6, 0.9, 1.0)]
    assert all(a < b for a, b in zip(vs, vs[1:])), vs
    # ③ m을 늘리면 감소 폭이 커진다 (같은 ρ)
    assert var_reduction(0.5, 8) < var_reduction(0.5, 4)
    # ④ 등상관 표본으로 항등식 확인
    g = np.random.default_rng(11)
    for rho in (0.0, 0.5, 0.9):
        cov = np.full((4, 4), rho) + np.eye(4) * (1 - rho)
        L = np.linalg.cholesky(cov)
        x = L @ g.standard_normal((4, 300_000))
        got = float(np.var(x.mean(axis=0), ddof=1))
        assert abs(got / var_reduction(rho) - 1.0) < 0.03, (rho, got)
    # ⑤ offset 인덱스가 단조 증가하고 달마다 하나씩
    d = pd.date_range("2020-01-01", "2020-06-30", freq="B").to_numpy()
    for off in range(4):
        ix = offset_reb_indices(d, 0, off)
        assert ix == sorted(ix) and len(ix) == 6, (off, ix)
    # ⑥ offset이 클수록 달 안에서 더 늦은 날
    a0 = offset_reb_indices(d, 0, 0)
    a3 = offset_reb_indices(d, 0, 3)
    assert all(x < y for x, y in zip(a0, a3))
    print("selftest: 6개 항목 통과 (분산비극단·단조·m효과·항등식·offset단조·offset순서)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--measure":
        cmd_measure()
    else:
        print(__doc__)
