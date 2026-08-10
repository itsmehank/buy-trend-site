"""H-019 — "N종목 중 k개 교체" 설계의 검출력 상한(bound).

핵심은 **스케일링 모형을 쓰지 않는 상한 논증**이다.

  짝당 필요 효과 = K · σ_δ / √(유효 짝-월 수)          (K = 2.801585)
  ⟹ 필요 짝-월 = (K · σ_δ / e_pair)²

`유효 짝-월 ≤ k · n` 은 **항상** 참이다(짝 사이에 양의 상관이 있으면 유효 수가
줄어들 뿐 늘지 않는다). 따라서 `k·n`(가장 유리한 경우)으로 계산해 미달이면
**어떤 스케일링 가정에서도 미달**이다. 모형 선택이 결론을 바꾸지 않는다.

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/swap_design_bound.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/swap_design_bound.py --sigma    # σ_δ 직접 측정
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/swap_design_bound.py --bound    # 프론티어

**--sigma / --bound 는 δ의 산포만 쓰고 평균·부호를 계산하지 않는다.**
사전등록 오염 방지를 위한 의도적 제약이다(H-015 §5.1과 동일 방식).
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading, regime

_N = NormalDist()
K_MDE = _N.inv_cdf(0.975) + _N.inv_cdf(0.80)      # 2.801585

# 베이스라인 B (H-017 §2.1과 동일)
LB_FAR, LB_NEAR = 252, 21
MIN_BARS = 250
DV_WIN, DV_TOP_FRAC = 60, 0.50
HOLD_N = 20                       # N
POOL_N = 90                       # 교체 후보 풀
SMA_BENCH = 200
K_PAIRS = 5                       # 짝 수 (교체 깊이)

MARKET_CFG = {"kr": {"min_price": 1000.0}, "us": {"min_price": 5.0}}

#: 외부 사전 지정 — 문헌이 보고하는 짝당 효과크기 (월 %p)
E_PAIR = {"연 6%p (보수적 하단)": 0.50, "연 12%p (관대한 상단)": 1.00}


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # PROTOCOL §1 — KR 거래정지 봉(open=high=low=0, close만 이월) OR 마스킹
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    far, near = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[LB_FAR:], near[LB_NEAR:] = c[:-LB_FAR], c[:-LB_NEAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        pret = near / far - 1.0
        rev = c / np.where(np.isfinite(near), near, np.nan) - 1.0        # 최근 1개월
        far6 = np.full_like(c, np.nan)
        far6[126:] = c[:-126]
        mom6 = near / far6 - 1.0
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    vol60 = pd.DataFrame(ret).rolling(60).std().to_numpy(float)
    # 상대거래량 RV = MA20(거래량)/MA250(거래량) — H-018의 신호
    vdf = piv["volume"]
    with np.errstate(all="ignore"):
        rv = (vdf.rolling(20).mean() / vdf.rolling(250).mean()).to_numpy(float)

    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close_ff": close.ffill().to_numpy(float), "close": c,
            "pret": pret, "rev": rev, "mom6": mom6, "vol60": vol60, "rv": rv,
            "dollar_vol": (close * piv["volume"]).to_numpy(float),
            "bars": close.notna().cumsum().to_numpy(float)}


def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).last().values
            if v >= start_i]


def ranked(P, i: int, cfg: dict) -> np.ndarray:
    px = P["close"][i]
    base = (np.isfinite(px) & (px >= cfg["min_price"])
            & (P["bars"][i] >= MIN_BARS) & np.isfinite(P["pret"][i]))
    if not base.any():
        return np.array([], dtype=int)
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dvm = np.nanmedian(dv, axis=0)
    univ = base & np.isfinite(dvm)
    if not univ.any():
        return np.array([], dtype=int)
    keep = max(int(univ.sum() * DV_TOP_FRAC), 1)
    thr = np.sort(dvm[univ])[::-1][keep - 1]
    univ &= dvm >= thr
    cand = np.flatnonzero(univ)
    tk = P["tickers"]
    return np.array(sorted(cand, key=lambda j: (-P["pret"][i][j], tk[j])), dtype=int)


#: 교체 신호 — 값이 **낮을수록** 빼고, **높을수록** 넣는다
SIGNALS = {
    "무작위(대조)": None,
    "단기반전(1M 역)": lambda P, i, j: -P["rev"][i][j],
    "저변동성": lambda P, i, j: -P["vol60"][i][j],
    "6개월 모멘텀": lambda P, i, j: P["mom6"][i][j],
    # H-018 — 상대거래량이 높을수록(과열) 탈락 대상이므로 부호를 뒤집는다
    "거래량과열(H-018)": lambda P, i, j: -P["rv"][i][j],
}


def measure_sigma(market: str, k: int = K_PAIRS) -> dict:
    """실제 교체 규칙이 만드는 짝의 δ 산포를 잰다. 평균·부호는 계산하지 않는다."""
    P = build_panel(market)
    cfg = MARKET_CFG[market]
    bmap = regime.bull_map(loading.load_bench(market), sma=SMA_BENCH)
    bull = np.array([bmap.get(d, False) for d in P["dates"]])
    start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                SMA_BENCH, LB_FAR, 1)
    rebs = [r for r in month_end_indices(P["dates"], start) if r + 1 < len(P["dates"])]
    c = P["close_ff"]
    rng = np.random.default_rng(20260811)

    out = {}
    n_bull = 0
    for name, fn in SIGNALS.items():
        deltas, per_month, n_bull = [], [], 0
        for a, b in zip(rebs[:-1], rebs[1:]):
            if not bull[a]:
                continue
            order = ranked(P, a, cfg)
            if len(order) < POOL_N:
                continue
            n_bull += 1
            top, pool = order[:HOLD_N], order[HOLD_N:POOL_N]
            if fn is None:
                drop = rng.choice(top, size=k, replace=False)
                add = rng.choice(pool, size=k, replace=False)
            else:
                sv_t = np.array([fn(P, a, j) for j in top], dtype=float)
                sv_p = np.array([fn(P, a, j) for j in pool], dtype=float)
                if not np.isfinite(sv_t).all() or np.isfinite(sv_p).sum() < k:
                    continue
                drop = top[np.argsort(sv_t)[:k]]                    # 신호 최저
                pf = pool[np.isfinite(sv_p)]
                sf = sv_p[np.isfinite(sv_p)]
                add = pf[np.argsort(-sf)[:k]]                       # 신호 최고
            with np.errstate(all="ignore"):
                r_old = c[b + 1][drop] / c[a + 1][drop] - 1.0
                r_new = c[b + 1][add] / c[a + 1][add] - 1.0
            d = (r_new - r_old) * 100.0
            d = d[np.isfinite(d)]
            if len(d) < k:
                continue
            deltas.append(d)
            per_month.append(d.sum())
        if not deltas:
            continue
        flat = np.concatenate(deltas)
        sig = float(np.std(flat, ddof=1))
        # 짝 간 상관: Var(Σδ) = k·σ²(1+(k−1)ρ) 에서 역산 (참고 지표)
        var_sum = float(np.var(np.array(per_month), ddof=1))
        rho = (var_sum / (k * sig ** 2) - 1.0) / (k - 1) if k > 1 and sig > 0 else np.nan
        out[name] = {"sigma": sig, "n_pairs": len(flat), "n_months": len(deltas),
                     "se_sigma": sig / np.sqrt(2 * (len(flat) - 1)), "rho": rho}
    out["_n_bull"] = n_bull
    return out


def need_pair_months(sigma: float, e_pair: float) -> float:
    """짝당 효과 e_pair 를 80% 검정력으로 잡는 데 필요한 **짝-월 수**."""
    return (K_MDE * sigma / e_pair) ** 2


def cmd_sigma():
    print("=" * 94)
    print("[H-019 σ_δ 직접 측정] 실제 교체 규칙이 만드는 짝의 산포 — 평균·부호 미계산")
    print(f"  베이스라인 B · 상위 {HOLD_N}종목 중 {K_PAIRS}개를 풀(21~{POOL_N}위)에서 교체")
    print("=" * 94)
    for m in ("kr", "us"):
        res = measure_sigma(m)
        print(f"\n── {m.upper()} ──  강세 달 {res['_n_bull']}")
        print(f"{'교체 신호':<18}{'σ_δ(%p/월)':>12}{'SE(σ)':>9}{'짝 수':>8}"
              f"{'달 수':>7}{'짝간 ρ':>9}")
        for name, r in res.items():
            if name.startswith("_"):
                continue
            print(f"{name:<18}{r['sigma']:>12.2f}{r['se_sigma']:>9.2f}"
                  f"{r['n_pairs']:>8}{r['n_months']:>7}{r['rho']:>9.3f}")
    print("\n※ 짝간 ρ 는 참고값이다. 판정은 ρ 를 쓰지 않는다 — §2.1의 상한 논증 참조.")


def cmd_bound():
    print("=" * 94)
    print("[H-019 검출력 상한] 필요 짝-월 = (K·σ_δ / e_pair)²  ·  가용 ≤ k × 강세달")
    print(f"  K = {K_MDE:.6f} (α=5%, 검정력 80%)")
    print("=" * 94)
    for m in ("kr", "us"):
        res = measure_sigma(m)
        n_bull = res["_n_bull"]
        sigs = {k: v["sigma"] for k, v in res.items() if not k.startswith("_")}
        lo, hi = min(sigs.values()), max(sigs.values())
        print(f"\n── {m.upper()} ──  강세 달 {n_bull} · σ_δ 범위 {lo:.2f} ~ {hi:.2f} %p")
        print(f"{'e_pair':<22}{'필요 짝-월(최소σ)':>18}{'필요(최대σ)':>14}"
              f"{'가용 k=5':>10}{'가용 k=20':>11}{'배율(최소σ)':>13}")
        for label, e in E_PAIR.items():
            n_lo, n_hi = need_pair_months(lo, e), need_pair_months(hi, e)
            av5, av20 = K_PAIRS * n_bull, HOLD_N * n_bull
            print(f"{label:<22}{n_lo:>18,.0f}{n_hi:>14,.0f}{av5:>10,}{av20:>11,}"
                  f"{n_lo / av5:>12.1f}배")
        # 결론을 뒤집는 e_pair
        for k, av in ((f"k={K_PAIRS}", K_PAIRS * n_bull), ("k=20", HOLD_N * n_bull)):
            flip = K_MDE * lo / np.sqrt(av)
            print(f"  결론을 뒤집는 e_pair ({k}, 최소 σ) : 월 {flip:.2f}%p = 연 {flip*12:.1f}%p")
    print("\n※ '가용'은 짝이 서로 독립일 때의 **상한**이다. 양의 상관이 있으면 더 줄어든다."
          "\n  따라서 이 비교로 미달이면 어떤 스케일링 가정에서도 미달이다.")


def selftest():
    # ① 상수
    assert abs(K_MDE - 2.801585) < 1e-6, K_MDE

    # ② need_pair_months 의 차원·단조성
    assert abs(need_pair_months(20.0, 0.5) - (K_MDE * 40) ** 2) < 1e-6
    assert need_pair_months(20.0, 0.5) > need_pair_months(20.0, 1.0)      # e 클수록 쉬움
    assert need_pair_months(30.0, 0.5) > need_pair_months(20.0, 0.5)      # σ 클수록 어려움

    # ③ e_pair 를 2배 하면 필요 짝-월은 정확히 1/4
    assert abs(need_pair_months(25.0, 1.0) / need_pair_months(25.0, 0.5) - 0.25) < 1e-12

    # ④ 소형 수치 예제 — 손계산과 일치
    #    σ=20, e=0.5 → (2.801585×20/0.5)² = 112.0634² = 12558.2
    v = need_pair_months(20.0, 0.5)
    assert abs(v - 12558.20) < 0.5, v

    # ⑤ 짝-월 → 짝당 필요 효과 역변환 왕복
    for sig in (15.0, 25.0, 35.0):
        for e in (0.3, 0.5, 1.0):
            n = need_pair_months(sig, e)
            assert abs(K_MDE * sig / np.sqrt(n) - e) < 1e-9

    # ⑥ **상한 논증의 핵심** — 짝 간 양의 상관은 유효 짝 수를 늘리지 못한다
    #    Var(Σδ) = k·σ²(1+(k−1)ρ) 이므로 k_eff = k/(1+(k−1)ρ) ≤ k  (ρ ≥ 0)
    for k in (2, 5, 20, 100):
        for rho in (0.0, 0.05, 0.3, 0.9):
            k_eff = k / (1 + (k - 1) * rho)
            assert k_eff <= k + 1e-12, (k, rho, k_eff)
        assert abs(k / (1 + (k - 1) * 0.0) - k) < 1e-12          # ρ=0 이면 등호

    # ⑦ 교체 짝 구성이 실제로 신호를 따르는가 (합성 데이터)
    P = {"rev": np.array([[0.5, 0.1, 0.9, 0.2]]),
         "vol60": np.zeros((1, 4)), "mom6": np.zeros((1, 4)),
         "rv": np.array([[1.0, 3.0, 0.5, 2.0]])}
    f = SIGNALS["단기반전(1M 역)"]
    v = [f(P, 0, j) for j in range(4)]
    assert np.argmin(v) == 2 and np.argmax(v) == 1   # 많이 오른 종목이 최저 → 탈락
    g = SIGNALS["거래량과열(H-018)"]
    w = [g(P, 0, j) for j in range(4)]
    assert np.argmin(w) == 1 and np.argmax(w) == 2   # RV 최고 종목이 최저값 → 탈락

    # ⑧ Var(Σδ) = k·σ²(1+(k−1)ρ) 항등식 — ⑥이 대입만 하므로 여기서 표본으로 확인
    rg = np.random.default_rng(5)
    for k, rho in ((5, 0.0), (5, 0.30), (20, 0.10)):
        cov = np.full((k, k), rho) + np.eye(k) * (1 - rho)
        L = np.linalg.cholesky(cov)
        x = (L @ rg.standard_normal((k, 200_000))) * 25.0
        want = k * 25.0 ** 2 * (1 + (k - 1) * rho)
        got = float(np.var(x.sum(axis=0), ddof=1))
        assert abs(got / want - 1.0) < 0.03, (k, rho, got, want)
        # ⑨ 스크립트가 쓰는 ρ 역산이 참값을 복원하는가
        sig = float(np.std(x.ravel(), ddof=1))
        back = (got / (k * sig ** 2) - 1.0) / (k - 1)
        assert abs(back - rho) < 0.02, (k, rho, back)

    print("selftest: 9개 항목 통과 (상수·차원단조·1/4법칙·손계산예제·왕복·"
          "상한부등식·신호방향·분산항등식·ρ역산)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--sigma":
        cmd_sigma()
    elif arg == "--bound":
        cmd_bound()
    else:
        print(__doc__)
