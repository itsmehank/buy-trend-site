"""통합 표본의 관측 종속성 — 실행된 판정 전건 재계산.

PROTOCOL §3은 KR·US 월별 차분을 **이어붙여** 판정한다. 그 SE를 `s/√n` 으로 내면
**n개 관측이 서로 독립**이라고 가정하는 것인데, 두 축에서 그 가정이 깨진다.

  (ㄱ) **시장 간(cross-section)** — 두 시장의 **같은 달**은 상관돼 있다.
  (ㄴ) **시계열(serial)** — 같은 시장의 이웃 달이 상관돼 있다.

**어느 한 추정량을 고르지 않는다.** 아래 6종을 전부 산출하고,
**판정이 갈리면 `측정 불가`로 기록**한다 (PROTOCOL §3 개정안).
추정량을 고르는 순간 "유리한 것을 골랐다"를 반증할 수 없기 때문이다.

  ① 달 클러스터   — (ㄱ)만
  ② 분기 클러스터 — (ㄱ) + 분기 내 (ㄴ)
  ③ 연 클러스터   — (ㄱ) + 연 내 (ㄴ)
  ④~⑥ Driscoll–Kraay L=3·6·12 — (ㄱ) + (ㄴ)을 HAC 로 동시에

**부호 규약 (2026-08-18 추가)** — `verdict()`·`robust_verdict()` 는 `direction`
인자를 받는다. **문헌이 예측하는 부호**(+1 또는 −1)이며 그 방향으로 유의해야 `채택`이다.
기본값 +1 은 H-020·H-022·H-025·H-026 처럼 문헌이 양수를 예측하는 경우다.
**문헌이 음수를 예측하는 가설(H-028 MAX)은 반드시 −1 을 넘겨야 한다** —
넘기지 않으면 원문 방향으로 나온 결과가 `기각`으로 뒤집힌다(게이트가 잡은 실제 결함).

`naive` 는 **비교용으로만 출력**하고 견고성 판정에 투표하지 않는다
(독립 가정이 깨진 것이 확인됐으므로 편향된 추정량이다).

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/pooled_clustering.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/pooled_clustering.py --run
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()

# 견고성 판정에 투표하는 추정량 — **사전 고정**. 결과를 보고 고르지 않는다.
VOTERS = ("달", "분기", "연", "DK L=3", "DK L=6", "DK L=12")


def _stack(series: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.concat([s.rename("v").to_frame().assign(mk=m) for m, s in series.items()])


def naive_se(series: dict) -> tuple[float, float]:
    df = _stack(series)
    return float(df["v"].mean()), float(df["v"].std(ddof=1) / np.sqrt(len(df)))


def cluster_se(series: dict, freq: str = "M") -> tuple[float, float, int]:
    """달/분기/연을 클러스터로 하는 1-way 로버스트 SE.

        SE = sqrt( G/(G−1) · Σ_g ( Σ_{i∈g} (x_i − x̄) )² ) / n

    `G/(G−1)` 은 표준 유한표본 보정(평균 추정량이라 (n−1)/(n−k) 항은 1이다).
    이것이 있어야 **클러스터가 전부 1개 관측일 때 naive 와 정확히 일치**한다.
    """
    df = _stack(series)
    n = len(df)
    mu = float(df["v"].mean())
    key = df.index if freq == "M" else df.index.asfreq(freq)
    g = (df["v"] - mu).groupby(key).sum()
    G = len(g)
    if G < 2:
        return mu, float("nan"), G
    return mu, float(np.sqrt(G / (G - 1) * (g ** 2).sum()) / n), G


def driscoll_kraay_se(series: dict, lag: int) -> tuple[float, float]:
    """시장 간 상관과 시계열 상관을 **동시에** 다룬다.

    각 달의 잔차를 시장 축으로 합쳐 `h_t` 를 만든 뒤 Newey–West(Bartlett)를 적용한다.
    lag=0 이면 달 클러스터와 같아진다(`--selftest` ⑧).
    """
    df = _stack(series)
    n = len(df)
    mu = float(df["v"].mean())
    h = (df["v"] - mu).groupby(df.index).sum().sort_index().to_numpy()
    T = len(h)
    S = float((h * h).sum())
    for l in range(1, lag + 1):
        S += 2.0 * (1.0 - l / (lag + 1.0)) * float((h[l:] * h[:-l]).sum())
    S *= T / (T - 1.0)                                  # 유한표본 보정
    return mu, float(np.sqrt(max(S, 0.0)) / n)


def inverse_variance(series: dict, use_rho: bool) -> tuple[float, float]:
    """역분산결합 — **클러스터 단위를 전혀 쓰지 않는** 독립 교차확인용.

    `use_rho=False` 는 두 시장 독립 가정(종전 견고성 통계의 방식),
    `True` 는 겹치는 달의 시장 간 상관을 공분산으로 반영한다.
    """
    ks = list(series)
    a, b = series[ks[0]], series[ks[1]]
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    w1 = (1 / v1) / (1 / v1 + 1 / v2)
    w2 = 1 - w1
    mu = w1 * a.mean() + w2 * b.mean()
    var = w1 ** 2 * v1 + w2 ** 2 * v2
    if use_rho:
        j = pd.concat({ks[0]: a, ks[1]: b}, axis=1).dropna()
        rho = j[ks[0]].corr(j[ks[1]])
        # 겹치는 구간만 공분산이 실린다
        var += 2 * w1 * w2 * rho * np.sqrt(v1 * v2) * (len(j) / np.sqrt(n1 * n2))
    return float(mu), float(np.sqrt(var))


def all_estimates(series: dict) -> dict:
    mu, nv = naive_se(series)
    n = len(_stack(series))
    out = {"mu": mu, "naive": nv, "n": n, "df_naive": n - 1}
    for tag, f in (("달", "M"), ("분기", "Q"), ("연", "Y")):
        _, se, G = cluster_se(series, f)
        out[tag] = se
        out[f"G_{tag}"] = G
        out[f"df_{tag}"] = G - 1                 # 클러스터 로버스트의 자유도
    for L in (3, 6, 12):
        _, se = driscoll_kraay_se(series, L)
        out[f"DK L={L}"] = se
        out[f"df_DK L={L}"] = out["G_달"] - 1     # DK 의 자유도 = 시점 수 − 1
    ks = list(series)
    j = pd.concat({m: series[m] for m in ks}, axis=1).dropna()
    out["rho"] = float(j[ks[0]].corr(j[ks[1]])) if len(j) > 2 else float("nan")
    out["overlap"] = len(j)
    for tag, ur in (("IV 독립", False), ("IV ρ반영", True)):
        m_, s_ = inverse_variance(series, ur)
        out[tag] = s_
        out[f"mu_{tag}"] = m_
    return out


def verdict(t: float, crit: float, direction: int = +1) -> str:
    """3분류. `direction` = **문헌이 예측하는 부호**(+1 또는 −1).

    문헌 방향으로 유의하면 `채택`, 반대 방향으로 유의하면 `기각`이다
    (PROTOCOL §3.1-7). 기본값 +1 은 문헌이 양수를 예측하는 경우로,
    H-020·H-022·H-025·H-026 이 전부 여기 해당한다.
    **문헌이 음수를 예측하는 가설(예: H-028 MAX)은 direction=−1 을 넘겨야 한다** —
    넘기지 않으면 원문 방향으로 나온 결과가 `기각`으로 뒤집힌다(게이트가 잡은 결함).
    """
    if not np.isfinite(t):
        return "산출 불가"
    if abs(t) <= crit:
        return "측정 불가"
    return "채택" if t * direction > 0 else "기각"


def _t_cdf(x: float, df: int) -> float:
    """t(df) 누적분포 — 밀도를 Simpson 적분한다 (scipy 의존 없음)."""
    if x < 0:
        return 1.0 - _t_cdf(-x, df)
    import math
    lc = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
          - 0.5 * math.log(df * math.pi))
    m = 4000                                   # 짝수
    h = x / m
    s = 0.0
    for i in range(m + 1):
        t = i * h
        w = 1 if i in (0, m) else (4 if i % 2 else 2)
        s += w * math.exp(lc - (df + 1) / 2 * math.log1p(t * t / df))
    return 0.5 + s * h / 3.0


def t_crit(df: int, family: int) -> float:
    """t(df) 참조분포의 Bonferroni 양측 임계. df<1 이면 무한대(판정 불가)."""
    if df < 1:
        return float("inf")
    p = 1 - 0.05 / (2 * family)
    lo, hi = 0.0, 4.0
    #  상한을 고정하면 꼬리가 두꺼운 저자유도(df=1 Cauchy)에서 **조용히 포화**한다.
    #  게이트 3차가 잡은 결함 — 도달할 때까지 넓힌다.
    while _t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError(f"t_crit 상한 초과 (df={df}, family={family})")
    for _ in range(200):                       # 이분법
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def robust_verdict(est: dict, crit: float, *, use_t: bool = False,
                   family: int = 2, direction: int = +1) -> tuple[str, dict, bool]:
    """VOTERS 전원이 같은 판정이면 그 판정, 하나라도 갈리면 **측정 불가**.

    `use_t=True` 면 추정량마다 **자기 자유도**의 t 임계를 쓴다
    (게이트 2차 지적: 분기 클러스터의 t 를 달 클러스터의 임계와 비교하면 안 된다).
    """
    vs = {}
    for k in VOTERS:
        c = t_crit(est[f"df_{k}"], family) if use_t else crit
        vs[k] = verdict(est["mu"] / est[k], c, direction)
    uniq = set(vs.values())
    agree = len(uniq) == 1
    return (uniq.pop() if agree else "측정 불가"), vs, agree


# ─────────────────────────────────────────────────────── 가설별 시계열 재구성

def series_broad_baseline() -> dict:
    """H-020(max) · H-022(expo) 및 H-023 측정 부품 — broad_baseline.py 재사용."""
    import broad_baseline as BB
    from btlib import loading, regime

    out: dict = {}
    for m in ("kr", "us"):
        cost = 0.0014 if m == "kr" else 0.0005
        P = BB.build_panel(m)
        cfg = BB.MARKET_CFG[m]
        bmap = regime.bull_map(loading.load_bench(m), sma=BB.SMA_BENCH)
        bull = np.array([bmap.get(d, False) for d in P["dates"]])
        start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= BB.MIN_BARS)),
                    BB.SMA_BENCH, BB.BETA_WIN, 1)
        rebs = [r for r in BB.month_end_indices(P["dates"], start)
                if r + 1 < len(P["dates"])]
        idx = pd.PeriodIndex(pd.to_datetime([P["dates"][a] for a in rebs[:-1]]),
                             freq="M")
        base = BB.run_arm(P, bull, cfg, None, cost)
        for part, label, _ in BB.PARTS:
            arm = BB.run_arm(P, bull, cfg, part, cost)
            d = np.asarray(arm["net"]) - np.asarray(base["net"])
            out.setdefault(part, {})[m] = pd.Series(d, index=idx[:len(d)])
    return out


def series_ma_timing() -> dict:
    """H-025 — 십분위 동일가중 MAP 월별 계열."""
    import ma_timing_deciles as MT

    out = {}
    for m in ("kr", "us"):
        P = MT.build_panel(m)
        ye = [i for i in MT.year_end_indices(P["dates"])
              if P["bars"][i].max() >= MT.MIN_BARS]
        segs, dseg = {j: [] for j in range(MT.N_DECILE)}, []
        for a, b in zip(ye[:-1], ye[1:]):
            decs = MT.form_deciles(P, a, m)
            if not decs or len(decs) != MT.N_DECILE or (b - a) < 5:
                continue
            for j, mem in enumerate(decs):
                segs[j].append(MT.decile_index(P, mem, a, b) if len(mem)
                               else np.zeros(b - a))
            dseg.append(P["dates"][a + 1:b + 1])
        dts = pd.to_datetime(np.concatenate(dseg))
        cols = []
        for j in range(MT.N_DECILE):
            r = np.concatenate(segs[j])
            tim, bh, _, _ = MT.ma_timing(r, MT.L_MAIN, MT.COST[m], None)
            cols.append(pd.Series(tim - bh, index=dts).resample("ME").sum() * 100)
        eq = pd.concat(cols, axis=1).mean(axis=1)
        out[m] = pd.Series(eq.to_numpy(), index=pd.PeriodIndex(eq.index, freq="M"))
    return out


def series_resid_deciles() -> dict:
    """H-026 — 잔차 모멘텀 D10−D1 월별 계열."""
    import residual_momentum_deciles as RM

    out = {}
    for m in ("kr", "us"):
        P = RM.build_panel(m)
        a = RM.run_arm(P, signal="resid")
        idx = pd.PeriodIndex(pd.to_datetime(P["dates"])[P["me"][a["months"]]], freq="M")
        out[m] = pd.Series(a["spread"], index=idx)
    return out


CASES = [
    ("H-020", "MAX 상위10% 제외", "max", 6, "측정 불가"),
    ("H-022", "실현분산 노출 조절", "expo", 6, "측정 불가"),   # 2026-08-12 하향 반영
    ("H-025", "MA 타이밍 × 변동성 십분위", "ma", 2, "기각"),
    ("H-026", "잔차 모멘텀 D10−D1", "resid", 2, "측정 불가"),
]


def cmd_run():
    print("=" * 104)
    print("[통합 표본의 관측 종속성] — 실행된 판정 전건 재계산")
    print("  추정량을 고르지 않는다. 6종 전부 산출하고 **갈리면 측정 불가**로 기록한다.")
    print("  naive 는 비교용 — 독립 가정이 깨졌으므로 견고성 판정에 투표하지 않는다.")
    print("=" * 104)
    bb = series_broad_baseline()
    src = {"max": bb["max"], "expo": bb["expo"],
           "ma": series_ma_timing(), "resid": series_resid_deciles()}
    changed = []
    for hid, label, key, family, prior in CASES:
        est = all_estimates(src[key])
        crit = _N.inv_cdf(1 - 0.05 / (2 * family))
        rv, vs, agree = robust_verdict(est, crit)
        print(f"\n── {hid} {label} ──")
        print(f"   n={est['n']} · 겹치는 달 {est['overlap']} · **시장 간 ρ = {est['rho']:+.4f}**"
              f" · 클러스터 수 G(달)={est['G_달']} G(분기)={est['G_분기']} G(연)={est['G_연']}")
        print(f"   평균 {est['mu']:+.4f} · 사전 임계 |t| > {crit:.3f} (가족 {family}칸)")
        print(f"   {'추정량':<10}{'SE':>9}{'t':>9}   {'판정(z)':<10}"
              f"{'df':>5}{'t(df)임계':>10}  판정(t분포)")
        print(f"   {'naive(참고)':<10}{est['naive']:>9.4f}"
              f"{est['mu']/est['naive']:>9.3f}   {verdict(est['mu']/est['naive'], crit):<10}")
        for k in VOTERS:
            df = est[f"df_{k}"]
            ct = t_crit(df, family)
            tv = est["mu"] / est[k]
            print(f"   {k:<10}{est[k]:>9.4f}{tv:>9.3f}   {vs[k]:<10}"
                  f"{df:>5}{ct:>10.3f}  {verdict(tv, ct)}")
        mark = "일치" if agree else "**갈림 → 측정 불가**"
        print(f"   → 6종 {mark} · **견고성 판정 = {rv}** (registry 기재: {prior})")
        rv_t, _, ag_t = robust_verdict(est, crit, use_t=True, family=family)
        print(f"   → t(df) 참조분포로 바꿔도: **{rv_t}** "
              f"({'일치' if ag_t else '갈림'}) — 판정 {'불변' if rv_t == rv else '**변동**'}")
        print(f"   [클러스터 단위 무관 교차확인] 역분산결합"
              f"  독립가정 t={est['mu_IV 독립']/est['IV 독립']:+.3f}"
              f"  · ρ반영 t={est['mu_IV ρ반영']/est['IV ρ반영']:+.3f}"
              f"  (임계 {crit:.3f})")
        if rv != prior:
            changed.append((hid, prior, rv))

    print(f"\n{'=' * 104}\n[참고] H-023이 '측정'만 한 부품 — 판정 대상이 아니다")
    import broad_baseline as BB
    for part, label, _ in BB.PARTS:
        if part in ("max", "expo"):
            continue
        e = all_estimates(bb[part])
        ts = " ".join(f"{e['mu']/e[k]:+.2f}" for k in VOTERS)
        print(f"   {label:<24} ρ={e['rho']:+.3f}  naive {e['mu']/e['naive']:+.2f} | {ts}")

    print(f"\n{'=' * 104}")
    if changed:
        for hid, a, b in changed:
            print(f"**판정 변경: {hid}  {a} → {b}**")
    else:
        print("판정 변경 없음")
    print("\n[관찰] ρ 는 가설마다 0.01~0.63으로 갈린다 — 일률 보정 계수를 쓸 수 없다.")
    print("  종목 단위 선별(H-020)은 ρ≈0, 지수 단위 조절(H-022)은 ρ≈0.63.")
    print("  **시장 전체를 보는 신호일수록 두 시장이 같이 움직여 상관이 커진다.**")


def selftest():
    ix = pd.period_range("2020-01", periods=6, freq="M")
    a = pd.Series([1.0, 2, 3, 4, 5, 6], index=ix)

    # ① 두 시장이 겹치지 않으면 달 클러스터 == naive (G/(G−1) 보정이 있어야 성립)
    b = pd.Series([1.0, 2, 3], index=pd.period_range("2030-01", periods=3, freq="M"))
    _, nv = naive_se({"kr": a, "us": b})
    _, cl, G = cluster_se({"kr": a, "us": b})
    assert G == 9, G
    assert abs(cl - nv) < 1e-12, (cl, nv)

    # ② 두 계열이 완전히 같으면 비 = sqrt((2G−1)/(G−1)) → G→∞ 에서 √2
    _, nv = naive_se({"kr": a, "us": a.copy()})
    _, cl, G = cluster_se({"kr": a, "us": a.copy()})
    assert G == 6 and abs((cl / nv) ** 2 - (2 * G - 1) / (G - 1)) < 1e-9, cl / nv
    big = pd.Series(np.random.default_rng(0).normal(size=400),
                    index=pd.period_range("2000-01", periods=400, freq="M"))
    _, nv = naive_se({"kr": big, "us": big.copy()})
    _, cl, _ = cluster_se({"kr": big, "us": big.copy()})
    assert abs(cl / nv - np.sqrt(2)) < 0.01, cl / nv

    # ③ 클러스터 안에서 **정확히 상쇄**되면 SE = 0
    #    (ρ=−1 만으로는 0이 아니다 — 기울기가 −1일 때만 상쇄된다)
    _, cl, _ = cluster_se({"kr": a, "us": (2 * a.mean() - a)})
    assert cl < 1e-12, cl
    _, cl2, _ = cluster_se({"kr": a, "us": (5 - 2 * a)})      # ρ=−1 이지만 상쇄 아님
    assert cl2 > 1e-3, cl2

    # ④ 평균·naive SE 는 단순 concat 과 같다
    o = pd.Series([2.0, 1, 4, 3, 6, 5], index=ix)
    mu, nv = naive_se({"kr": a, "us": o})
    allv = np.concatenate([a.to_numpy(), o.to_numpy()])
    assert abs(mu - allv.mean()) < 1e-12
    assert abs(nv - allv.std(ddof=1) / np.sqrt(12)) < 1e-12

    # ⑤ verdict 3분류 + 견고성 규칙
    assert verdict(3.0, 2.24) == "채택" and verdict(-3.0, 2.24) == "기각"
    assert verdict(1.0, 2.24) == "측정 불가"
    est = {"mu": 1.0, **{k: 0.2 for k in VOTERS}}             # 전부 t=5 → 채택
    rv, _, ag = robust_verdict(est, 2.24)
    assert rv == "채택" and ag
    est["DK L=6"] = 1.0                                       # 한 칸만 t=1 → 갈림
    rv, _, ag = robust_verdict(est, 2.24)
    assert rv == "측정 불가" and not ag

    # ⑤-b **판정 부호 규약** — 문헌 예측 방향으로 유의해야 채택이다
    assert verdict(+3.0, 2.24, +1) == "채택" and verdict(-3.0, 2.24, +1) == "기각"
    assert verdict(-3.0, 2.24, -1) == "채택" and verdict(+3.0, 2.24, -1) == "기각"
    assert verdict(1.0, 2.24, -1) == "측정 불가"
    est5 = {"mu": -1.0, **{k: 0.2 for k in VOTERS}, **{f"df_{k}": 50 for k in VOTERS}}
    assert robust_verdict(est5, 2.24, direction=-1)[0] == "채택"   # 음수 문헌
    assert robust_verdict(est5, 2.24, direction=+1)[0] == "기각"   # 양수 문헌
    #    기존 4건(문헌 양수)은 기본값으로 동작이 바뀌지 않는다
    assert robust_verdict(est5, 2.24)[0] == "기각"

    # ⑥ 임계값
    assert abs(_N.inv_cdf(1 - 0.05 / (2 * 6)) - 2.6383) < 5e-4
    assert abs(_N.inv_cdf(1 - 0.05 / (2 * 2)) - 2.2414) < 5e-4

    # ⑦ **정렬 검증** — series_* 의 인덱스가 어긋나면 ρ 가 무너진다.
    #    한 시장을 1개월 밀면 상관이 사라지는지 확인한다(피검 함수의 유일한 약점).
    rng = np.random.default_rng(3)
    common = rng.normal(size=120)
    s1 = pd.Series(common + rng.normal(0, 0.3, 120),
                   index=pd.period_range("2010-01", periods=120, freq="M"))
    s2 = pd.Series(common + rng.normal(0, 0.3, 120), index=s1.index)
    e_ok = all_estimates({"kr": s1, "us": s2})
    e_bad = all_estimates({"kr": s1, "us": pd.Series(s2.to_numpy(), index=s1.index + 1)})
    assert e_ok["rho"] > 0.8, e_ok["rho"]
    assert abs(e_bad["rho"]) < 0.3, e_bad["rho"]
    assert e_ok["달"] > e_bad["달"], (e_ok["달"], e_bad["달"])

    # ⑧ Driscoll–Kraay: lag=0 이면 달 클러스터와 같다 (보정 계수까지)
    _, dk0 = driscoll_kraay_se({"kr": s1, "us": s2}, 0)
    _, cl0, G0 = cluster_se({"kr": s1, "us": s2}, "M")
    assert abs(dk0 - cl0) < 1e-12, (dk0, cl0, G0)
    #    양의 시계열 상관을 넣으면 DK SE 가 커진다
    ar = np.zeros(200)
    for i in range(1, 200):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    sa = pd.Series(ar, index=pd.period_range("2000-01", periods=200, freq="M"))
    _, d0 = driscoll_kraay_se({"kr": sa, "us": sa.copy()}, 0)
    _, d6 = driscoll_kraay_se({"kr": sa, "us": sa.copy()}, 6)
    assert d6 > d0 * 1.5, (d0, d6)

    # ⑨ **클러스터 단위가 실제로 다른 그룹을 쓰는가** (게이트 2차 변이 M2가 뚫은 자리).
    #    12개월 계열이면 G(달)=12 · G(분기)=4 · G(연)=1 이어야 한다.
    y1 = pd.Series(np.arange(12.0), index=pd.period_range("2021-01", periods=12, freq="M"))
    _, _, gm = cluster_se({"kr": y1, "us": y1.iloc[:0]}, "M")
    _, _, gq = cluster_se({"kr": y1, "us": y1.iloc[:0]}, "Q")
    _, seY, gy = cluster_se({"kr": y1, "us": y1.iloc[:0]}, "Y")
    assert (gm, gq, gy) == (12, 4, 1), (gm, gq, gy)
    assert not np.isfinite(seY), seY              # G<2 → 산출 불가
    assert verdict(float("nan"), 2.0) == "산출 불가"
    #    산출 불가가 섞이면 견고성 규칙이 자동으로 '갈림'을 낸다
    est9 = {"mu": 1.0, **{k: 0.2 for k in VOTERS}, **{f"df_{k}": 50 for k in VOTERS}}
    est9["연"] = float("nan")
    assert robust_verdict(est9, 2.24)[0] == "측정 불가"

    # ⑩ **DK 커널이 실제로 Bartlett 인가** (변이 M5가 뚫은 자리).
    #    절단커널(가중 1)과 다른 값이 나와야 한다.
    _, dk_b = driscoll_kraay_se({"kr": sa, "us": sa.copy()}, 6)
    df10 = _stack({"kr": sa, "us": sa.copy()})
    h10 = (df10["v"] - df10["v"].mean()).groupby(df10.index).sum().sort_index().to_numpy()
    T10 = len(h10)
    S10 = float((h10 * h10).sum())
    for l in range(1, 7):
        S10 += 2.0 * float((h10[l:] * h10[:-l]).sum())       # 가중 1 = 절단커널
    trunc = float(np.sqrt(max(S10 * T10 / (T10 - 1.0), 0.0)) / len(df10))
    assert abs(dk_b - trunc) > 1e-6, (dk_b, trunc)

    # ⑪ t 임계 — **닫힌 형태와 대조**한다 (scipy 없이 구현했으므로).
    import math
    for fam in (1, 2, 6, 12):
        p = 1 - 0.05 / (2 * fam)
        #  df=1 은 Cauchy: 분위수 = tan(π(p−0.5))
        want1 = math.tan(math.pi * (p - 0.5))
        got1 = t_crit(1, fam)
        assert abs(got1 / want1 - 1) < 1e-6, (fam, got1, want1)
        #  df=2 는 F(x)=0.5+x/(2√(2+x²)) → x = sqrt(8u²/(1−4u²)), u=p−0.5
        u = p - 0.5
        want2 = math.sqrt(8 * u * u / (1 - 4 * u * u))
        got2 = t_crit(2, fam)
        assert abs(got2 / want2 - 1) < 1e-6, (fam, got2, want2)
    #    단조성 · df→∞ 에서 z 수렴 · df<1 이면 무한대
    assert t_crit(55, 6) > t_crit(163, 6) > _N.inv_cdf(1 - 0.05 / 12), (
        t_crit(55, 6), t_crit(163, 6))
    assert abs(t_crit(10_000_000, 2) - _N.inv_cdf(1 - 0.05 / 4)) < 1e-3
    assert not np.isfinite(t_crit(0, 2))

    # ⑪-b **자유도 배정** — 2차 부결 사유의 핵심 축인데 검증이 없었다(게이트 3차 지적).
    e11 = all_estimates({"kr": s1, "us": s2})
    for tag in ("달", "분기", "연"):
        assert e11[f"df_{tag}"] == e11[f"G_{tag}"] - 1, tag
    for L in (3, 6, 12):
        assert e11[f"df_DK L={L}"] == e11["G_달"] - 1, L      # DK 자유도 = 시점 수 − 1
    assert e11["df_naive"] == e11["n"] - 1
    assert e11["df_분기"] < e11["df_달"], (e11["df_분기"], e11["df_달"])

    # ⑫ 역분산결합 — ρ>0 이면 SE 가 커진다 (클러스터 단위와 무관한 교차확인)
    _, iv0 = inverse_variance({"kr": s1, "us": s2}, False)
    _, iv1 = inverse_variance({"kr": s1, "us": s2}, True)
    assert iv1 > iv0, (iv0, iv1)

    print("selftest: 12개 항목 통과 (비겹침=naive · 완전동일=√2배 · 상쇄=0 · "
          "평균/naive 일치 · 3분류+견고성규칙+부호규약 · 임계값 · **정렬** · DK(lag0=클러스터·AR증가) · "
          "**클러스터단위 분리+G<2** · **DK커널=Bartlett** · **t임계(닫힌형태 Cauchy·df2)** · "
          "**자유도 배정** · 역분산결합)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.path.insert(0, "docs/analysis/backtests/scripts")
    if arg == "--selftest":
        selftest()
    elif arg == "--run":
        cmd_run()
    else:
        print(__doc__)
