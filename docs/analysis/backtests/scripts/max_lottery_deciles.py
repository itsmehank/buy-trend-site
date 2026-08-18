"""H-028 — MAX(복권형) 십분위. **Bali·Cakici·Whitelaw (2011) 원문 형태 그대로.**

원문(JFE 99(2), pp.427-446) 식 (2):
    MAX_{i,t} = max(R_{i,d}),  d = 1,…,D_t      ← **역월 안의 최대 일간수익률**

  · 매월 MAX 로 십분위. D1 = 최저 · D10 = 최고. 보유 1개월.
  · D10 − D1 zero-investment. 원문은 VW·EW 둘 다 보고 — **우리는 EW만**(시총 없음).
  · 레짐 필터 없음(원문에 없음).
  · 원문 Table 1 EW: 10−1 = **−0.65%p/월 (t=−1.83)**  ← 이 테스트의 비교 대상

판정은 PROTOCOL §3 개정판(2026-08-12) — **6종 추정량이 갈리면 측정 불가**.

  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/max_lottery_deciles.py --selftest | --power | --run

**--power 는 SE·검정력·비용만 출력한다. 평균·t·부호는 출력 경로가 없다.**
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading

sys.path.insert(0, "docs/analysis/backtests/scripts")

_N = NormalDist()

N_DECILE = 10
MIN_DAYS = 15                   # 월 안 유효 일간수익률 최소 개수 (§2.4 — 이 문서가 정함)
DV_WIN = 20
DV_MIN = {"kr": None, "us": 2e6}
PRICE_FLOOR = {"kr": 1000.0, "us": 1.0}
COST = {"kr": 0.0014, "us": 0.0005}
CAP_N = {"kr": 400, "us": 500}
CAP_WIN = 60
SMA_BENCH = 200

# 원문 Table 1 (EW, 월 %) — 사전 고정
LIT_EW_DECILES = (1.29, 1.45, 1.55, 1.55, 1.49, 1.49, 1.37, 1.32, 1.04, 0.64)
LIT_EW_SPREAD = -0.65           # 10−1, t=−1.83
LIT_VW_SPREAD = -1.03           # 참고 (재현 불가 — 시총 없음)

LIT_DIR = -1                    # **문헌이 예측하는 부호** — 음수 유의면 채택 (§3.1)
FAMILY = 2
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414


# ────────────────────────────────────────────────────────────── 패널

def month_key(dates) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(dates), freq="M")


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)
    bad = (piv["close"] <= 0) | (piv["open"] <= 0)             # PROTOCOL §1
    for k in piv:
        piv[k] = piv[k].mask(bad)
    close = piv["close"]
    cl = close.to_numpy(float)

    dv_d = close * piv["volume"]
    pit_dv = dv_d.rolling(DV_WIN, min_periods=DV_WIN).mean().to_numpy(float)
    cap_dv = dv_d.rolling(CAP_WIN, min_periods=CAP_WIN).median().to_numpy(float)

    # 일간수익률 — 월 첫 거래일은 **직전 월 마지막 거래일 종가** 대비 (§2.3)
    dret = np.full_like(cl, np.nan)
    dret[1:] = cl[1:] / cl[:-1] - 1.0

    mk = month_key(idx)
    me = pd.Series(np.arange(len(idx)), index=mk).groupby(level=0).last().to_numpy()
    me = me[:-1]                              # 마지막 역월은 부분 월 → 제외 (§2.3)
    months = pd.PeriodIndex(mk[me])

    pm = cl[me]
    mret = np.full_like(pm, np.nan)
    mret[1:] = pm[1:] / pm[:-1] - 1.0         # 월수익 (보유 수익용)

    # 역월별 MAX 와 유효 일수
    codes = mk.astype("period[M]")
    mx, nd = [], []
    for p in months:
        sel = np.flatnonzero(codes == p)
        seg = dret[sel]
        with np.errstate(all="ignore"):
            mx.append(np.nanmax(seg, axis=0) if len(seg) else np.full(cl.shape[1], np.nan))
        nd.append(np.isfinite(seg).sum(axis=0))

    b = loading.load_bench(market).set_index("date")["close"]
    b = b.reindex(pd.to_datetime(idx).date if not isinstance(idx[0], pd.Timestamp)
                  else idx).ffill()
    bv = b.to_numpy(float)
    sma = pd.Series(bv).rolling(SMA_BENCH, min_periods=SMA_BENCH).mean().to_numpy()

    return {"tickers": np.asarray(cols, dtype=str), "close": cl, "dret": dret,
            "me": me, "months": months, "n_days": len(idx), "codes": codes,
            "pm": pm, "mret": mret, "max": np.asarray(mx, dtype=float),
            "ndays": np.asarray(nd, dtype=float),
            "pit_dv": pit_dv[me], "cap_dv": cap_dv[me],
            "bull": (bv[me] > sma[me]), "market": market}


def max_n(P: dict, m: int, n: int) -> np.ndarray:
    """원문 Table 2 의 MAX(N) — 그달 **상위 N개** 일간수익률의 평균. N=1이면 MAX."""
    sel = np.flatnonzero(P["codes"] == P["months"][m])
    seg = P["dret"][sel]
    out = np.full(seg.shape[1], np.nan)
    for j in range(seg.shape[1]):
        v = seg[:, j][np.isfinite(seg[:, j])]
        if len(v) >= n:
            out[j] = np.sort(v)[::-1][:n].mean()
    return out


# ────────────────────────────────────────────────────────── 적격·십분위

def eligible(P: dict, m: int, *, cap: bool = False, min_days: int = MIN_DAYS,
             score: np.ndarray | None = None, use_floor: bool = True) -> np.ndarray:
    """`use_floor=False` 는 §3.2의 '가격 하한 없음' 민감도 — 원문에 없는
    저장소 관행(§2.2 이탈 5)을 뺐을 때를 본다. **결과를 보기 전에 고정했다.**
    """
    mk = P["market"]
    sc = P["max"][m] if score is None else score
    ok = np.isfinite(sc) & (P["ndays"][m] >= min_days)
    px = P["pm"][m]
    ok &= np.isfinite(px)
    if use_floor:
        ok &= px >= PRICE_FLOOR[mk]
    if DV_MIN[mk] is not None:
        dv = P["pit_dv"][m]
        ok &= np.isfinite(dv) & (dv >= DV_MIN[mk])
    if cap:
        dv = P["cap_dv"][m]
        ok &= np.isfinite(dv)
        cand = np.flatnonzero(ok)
        if len(cand) > CAP_N[mk]:
            thr = np.sort(dv[cand])[::-1][CAP_N[mk] - 1]
            ok &= dv >= thr
    return ok


def split_deciles(P: dict, score: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    """MAX **오름차순**(동점은 티커 오름차순) 10등분. [0]=D1(최저) … [9]=D10(최고)."""
    cand = np.flatnonzero(ok)
    if len(cand) < N_DECILE * 2:
        return []
    tk = P["tickers"]
    order = np.array(sorted(cand, key=lambda j: (score[j], tk[j])), dtype=int)
    return [np.asarray(g, dtype=int) for g in np.array_split(order, N_DECILE)]


# ─────────────────────────────────────────────────────────── 시뮬레이션

def _hold_return(P: dict, m: int, mem: np.ndarray, exec_lag: int) -> float:
    if len(mem) == 0:
        return 0.0
    if exec_lag == 0:
        r = P["mret"][m + 1][mem]
    else:
        i0, i1 = P["me"][m] + exec_lag, P["me"][m + 1] + exec_lag
        if i1 >= P["n_days"]:
            return np.nan
        c = P["close"]
        r = c[i1][mem] / c[i0][mem] - 1.0
    r = r[np.isfinite(r)]
    return float(r.mean()) if len(r) else 0.0


def _turn_cost(prev, mem, drift, cost):
    w_new = {int(j): 1.0 / len(mem) for j in mem} if len(mem) else {}
    if prev is None:
        drifted = {}
    else:
        tmp = {}
        for j, w in prev.items():
            g = drift[j]
            tmp[j] = w * (1.0 + (g if np.isfinite(g) else 0.0))
        s = sum(tmp.values())
        drifted = {j: v / s for j, v in tmp.items()} if s > 0 else {}
    keys = set(w_new) | set(drifted)
    turn = sum(abs(w_new.get(j, 0.0) - drifted.get(j, 0.0)) for j in keys)
    return cost * turn, w_new


def run_arm(P: dict, *, n_top: int = 1, cap: bool = False, exec_lag: int = 0,
            min_days: int = MIN_DAYS, use_floor: bool = True) -> dict:
    n_m = len(P["me"])
    cost = COST[P["market"]]
    prev = [None] * N_DECILE
    rows, gross_rows, cost_rows, turn_rows = [], [], [], []
    uni_rows, bull_rows, ndec, elig_n, months, maxavg = [], [], [], [], [], []
    # m=0 은 mret 이 NaN(기준월 없음)이라 시작 인덱스를 1로 둔다.
    # 판정 월 1개를 보수적으로 버리는 선택이며 §5.1 의 145/177 이 이 기준이다.
    for m in range(1, n_m - 1):
        sc = P["max"][m] if n_top == 1 else max_n(P, m, n_top)
        ok = eligible(P, m, cap=cap, min_days=min_days, score=sc, use_floor=use_floor)
        decs = split_deciles(P, sc, ok)
        if not decs:
            continue
        drift = np.nan_to_num(P["mret"][m], nan=0.0)
        gro = np.zeros(N_DECILE)
        cst = np.zeros(N_DECILE)
        turn = 0.0
        for d in range(N_DECILE):
            gro[d] = _hold_return(P, m, decs[d], exec_lag)
            cc, prev[d] = _turn_cost(prev[d], decs[d], drift, cost)
            cst[d] = cc
            if d in (0, N_DECILE - 1) and cost > 0:
                turn += cc / cost / 2.0 / 2.0        # 두 다리의 **평균** 편도 회전율
        uni = np.flatnonzero(ok)
        rows.append((gro - cst) * 100.0)
        gross_rows.append(gro * 100.0)
        cost_rows.append(cst * 100.0)
        turn_rows.append(turn)
        uni_rows.append(_hold_return(P, m, uni, exec_lag) * 100.0)
        bull_rows.append(bool(P["bull"][m]))
        ndec.append(np.mean([len(g) for g in decs]))
        elig_n.append(len(uni))
        maxavg.append([float(np.nanmean(sc[g])) * 100 for g in decs])
        months.append(m)
    if not rows:
        return {}
    R, G, C = np.asarray(rows), np.asarray(gross_rows), np.asarray(cost_rows)
    # zero-investment: 양 다리 모두 비용 차감
    spread = (G[:, N_DECILE - 1] - G[:, 0]) - (C[:, N_DECILE - 1] + C[:, 0])
    return {"dec": R, "gross": G, "cost": C, "spread": spread,
            "uni": np.asarray(uni_rows), "bull": np.asarray(bull_rows),
            "months": np.asarray(months), "turn": np.asarray(turn_rows),
            "maxavg": np.asarray(maxavg),
            "ndec": float(np.mean(ndec)), "elig": float(np.mean(elig_n)),
            "n": len(rows)}


def series_for(P: dict, **kw) -> pd.Series:
    a = run_arm(P, **kw)
    return pd.Series(a["spread"], index=P["months"][a["months"]])


# ────────────────────────────────────────────────────────────── 통계

def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def spearman(x, y) -> float:
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def cagr(monthly_pct) -> float:
    g = np.prod(1.0 + np.asarray(monthly_pct, float) / 100.0)
    return float(g ** (12.0 / len(monthly_pct)) - 1.0) * 100.0


def _pc():
    import pooled_clustering as PC
    return PC


# ────────────────────────────────────────────────────────────── 명령

def _panels():
    return {m: build_panel(m) for m in ("kr", "us")}


def cmd_power():
    PC = _pc()
    print("=" * 100)
    print("[H-028] 사전 검출력 — **SE·검정력·비용만**. 평균·t·부호는 출력 경로가 없다.")
    print(f"  가족 {FAMILY}칸 Bonferroni → |t| > {K_CRIT:.4f}"
          f" · 문헌 효과(EW) 월 {LIT_EW_SPREAD}%p")
    print("  검정력은 PROTOCOL §3 개정판대로 **6종 중 SE 최대(= 최저 검정력)** 기준.")
    print("=" * 100)
    ser = {}
    for mk, P in _panels().items():
        a = run_arm(P)
        ser[mk] = pd.Series(a["spread"], index=P["months"][a["months"]])
        cs = a["cost"]
        cm = float(cs[:, N_DECILE - 1].mean() + cs[:, 0].mean())
        print(f"\n── {mk.upper()} ── n={a['n']}개월 · 적격 평균 {a['elig']:.0f}종목"
              f" · 십분위 {a['ndec']:.1f}종목 · 회전율 {a['turn'].mean()*100:.1f}%/월")
        print(f"   스프레드 월 비용 {cm:.4f}%p (연 {cm*12:.2f}%p)"
              f" · 문헌 효과의 {abs(cm/LIT_EW_SPREAD)*100:.0f}% 잠식")
        se = PC.naive_se({"x": ser[mk]})[1]
        print(f"   naive SE {se:.4f}%p (참고)")
    est = PC.all_estimates(ser)
    ses = {k: est[k] for k in PC.VOTERS}
    worst = max(ses, key=lambda k: ses[k])
    d = abs(LIT_EW_SPREAD) / est[worst]
    pw = _N.cdf(d - K_CRIT) + _N.cdf(-d - K_CRIT)
    print(f"\n[통합] n={est['n']}개월 · 겹치는 달 {est['overlap']} · ρ={est['rho']:+.4f}")
    print(f"   G(달)={est['G_달']} G(분기)={est['G_분기']} G(연)={est['G_연']}")
    for k in PC.VOTERS:
        print(f"   {k:<10} SE {est[k]:.4f}%p")
    print(f"   naive      SE {est['naive']:.4f}%p (판정에 투표하지 않음)")
    print(f"\n   **SE 최대 = {worst} ({est[worst]:.4f}) → 검정력 = {pw*100:.1f}%**")
    print(f"   (naive 기준이었다면 "
          f"{(_N.cdf(abs(LIT_EW_SPREAD)/est['naive']-K_CRIT)+_N.cdf(-abs(LIT_EW_SPREAD)/est['naive']-K_CRIT))*100:.1f}%)")


def cmd_run():
    PC = _pc()
    print("=" * 104)
    print("[H-028] MAX 십분위 — Bali·Cakici·Whitelaw (2011) 원문 형태")
    print("  역월 최대 일간수익률 · 십분위 · D10−D1 zero-investment · 동일가중 · 레짐 없음")
    print(f"  판정: 통합 표본 · 가족 {FAMILY}칸 → |t| > {K_CRIT:.4f}"
          f" · **6종 갈리면 측정 불가**(PROTOCOL §3)")
    print(f"  문헌 예측(EW): D10−D1 = {LIT_EW_SPREAD}%p/월 (t=−1.83) · **음수면 채택**")
    print("=" * 104)
    P = _panels()
    ser, arms = {}, {}
    for mk in ("kr", "us"):
        a = run_arm(P[mk])
        arms[mk] = a
        ser[mk] = pd.Series(a["spread"], index=P[mk]["months"][a["months"]])
        mm = P[mk]["months"][a["months"]]
        print(f"\n{'─'*104}\n── {mk.upper()} ── n={a['n']}개월 "
              f"({mm[0]}~{mm[-1]}) · 적격 {a['elig']:.0f}종목 · 십분위 {a['ndec']:.1f}종목")
        print(f"{'십분위':>7}{'월수익':>10}{'연환산':>10}{'평균 MAX':>11}{'원문 EW':>10}")
        for d in range(N_DECILE):
            print(f"{'D'+str(d+1):>7}{a['dec'][:,d].mean():>+9.3f}%"
                  f"{a['dec'][:,d].mean()*12:>+9.2f}%"
                  f"{a['maxavg'][:,d].mean():>10.2f}%{LIT_EW_DECILES[d]:>9.2f}%")
        s = a["spread"]
        ann, vol = s.mean() * 12, s.std(ddof=1) * np.sqrt(12)
        print(f"  D10−D1  월 {s.mean():+.4f}%p · 연 {ann:+.2f}%p · naive t={tstat(s):+.2f}"
              f" · 변동성 {vol:.2f}% · Sharpe {ann/vol:+.2f} · P(>0) {(s>0).mean()*100:.0f}%")
        print(f"  십분위 순위상관 ρ = {spearman(np.arange(N_DECILE), a['dec'].mean(axis=0)):+.3f}"
              f"  (원문 EW 표의 ρ = {spearman(np.arange(N_DECILE), LIT_EW_DECILES):+.3f})")
        print(f"  회전율 {a['turn'].mean()*100:.1f}%/월 · D1 CAGR {cagr(a['dec'][:,0]):+.2f}%"
              f" · D10 CAGR {cagr(a['dec'][:,N_DECILE-1]):+.2f}%"
              f" · 유니버스 CAGR {cagr(a['uni']):+.2f}%")

    est = PC.all_estimates(ser)
    rv, vs, agree = PC.robust_verdict(est, K_CRIT, direction=LIT_DIR)
    print(f"\n{'='*104}\n[① 통합 표본 판정 — D10 − D1]")
    print(f"  n={est['n']}개월 · 겹치는 달 {est['overlap']} · ρ={est['rho']:+.4f}"
          f" · 월 {est['mu']:+.4f}%p · 연 {est['mu']*12:+.2f}%p")
    print(f"  {'추정량':<10}{'SE':>9}{'t':>9}   판정")
    print(f"  {'naive(참고)':<10}{est['naive']:>9.4f}{est['mu']/est['naive']:>9.3f}"
          f"   {PC.verdict(est['mu']/est['naive'], K_CRIT, LIT_DIR)}")
    for k in PC.VOTERS:
        print(f"  {k:<10}{est[k]:>9.4f}{est['mu']/est[k]:>9.3f}   {vs[k]}")
    print(f"  → 6종 {'일치' if agree else '**갈림**'} · **판정 = {rv}**"
          f"   (문헌 예측 {LIT_EW_SPREAD}%p — **음수 유의면 채택**)")
    lo = est["mu"] - 1.96 * est[max(PC.VOTERS, key=lambda k: est[k])]
    hi = est["mu"] + 1.96 * est[max(PC.VOTERS, key=lambda k: est[k])]
    print(f"  95% CI(SE 최대 기준) = [{lo:+.4f}, {hi:+.4f}] %p/월"
          f" · 0 포함? {'예' if lo<=0<=hi else '아니오'}"
          f" · 문헌값({LIT_EW_SPREAD}) 포함? {'예' if lo<=LIT_EW_SPREAD<=hi else '아니오'}")

    gser = {mk: pd.Series(arms[mk]["gross"][:, N_DECILE - 1] - arms[mk]["gross"][:, 0],
                          index=P[mk]["months"][arms[mk]["months"]]) for mk in ("kr", "us")}
    ge = PC.all_estimates(gser)
    grv, gvs, gag = PC.robust_verdict(ge, K_CRIT, direction=LIT_DIR)
    print(f"\n[병기] **비용 전(gross) D10−D1** — 원문은 비용을 고려하지 않으므로"
          f" 원문과 직접 비교 가능한 양이다 (판정에는 쓰지 않는다)")
    for mk in ("kr", "us"):
        g = gser[mk].to_numpy()
        print(f"  {mk.upper()} 월 {g.mean():+.4f}%p · 연 {g.mean()*12:+.2f}%p"
              f" · naive t={tstat(g):+.2f}")
    print(f"  통합 월 {ge['mu']:+.4f}%p · 연 {ge['mu']*12:+.2f}%p"
          f" · 6종 t {min(ge['mu']/ge[k] for k in PC.VOTERS):+.2f}"
          f" ~ {max(ge['mu']/ge[k] for k in PC.VOTERS):+.2f}"
          f" → {'일치' if gag else '갈림'} · {grv}   (문헌 {LIT_EW_SPREAD}%p)")

    print(f"\n[② 십분위 순위상관 ρ — 문헌 예측: 음수]")
    for mk in ("kr", "us"):
        print(f"  {mk.upper()} ρ = {spearman(np.arange(N_DECILE), arms[mk]['dec'].mean(axis=0)):+.3f}")
    print(f"  원문 EW 표 ρ = {spearman(np.arange(N_DECILE), LIT_EW_DECILES):+.3f}")

    print(f"\n[병기·**오염**] 롱온리 D1 − 유니버스 (레짐 적용)")
    lo_ser = {}
    for mk in ("kr", "us"):
        a = arms[mk]
        v = np.where(a["bull"], a["dec"][:, 0] - a["uni"], 0.0)
        lo_ser[mk] = pd.Series(v, index=P[mk]["months"][a["months"]])
        print(f"  {mk.upper()} 월 {v.mean():+.4f}%p · naive t={tstat(v):+.2f}")
    pl = np.concatenate([lo_ser[m].to_numpy() for m in ("kr", "us")])
    print(f"  통합 월 {pl.mean():+.4f}%p · naive t={tstat(pl):+.2f}   ← 문서 §7-1 참조")

    print(f"\n{'='*104}\n[견고성 — 병기, 판정에 쓰지 않는다]")
    print(f"{'변형':<30}{'통합 월':>11}{'연':>9}{'naive t':>10}{'n':>6}")
    for name, kw in (("MAX(2) 상위2 평균 (원문 T2)", dict(n_top=2)),
                     ("MAX(3) 상위3 평균", dict(n_top=3)),
                     ("MAX(5) 상위5 평균", dict(n_top=5)),
                     ("T+1 종가 체결", dict(exec_lag=1)),
                     (f"유니버스 캡 {CAP_N['kr']}/{CAP_N['us']}", dict(cap=True)),
                     ("최소 유효일 10일", dict(min_days=10)),
                     ("최소 유효일 20일", dict(min_days=20)),
                     ("**가격 하한 없음** (원문에 없는 필터 제거)", dict(use_floor=False))):
        segs = [series_for(P[mk], **kw) for mk in ("kr", "us")]
        p = np.concatenate([s.to_numpy() for s in segs])
        print(f"{name:<30}{p.mean():>+10.4f}%{p.mean()*12:>+8.2f}%{tstat(p):>10.2f}{len(p):>6}")

    for mk in ("kr", "us"):
        w = loading.staleness_warning(loading.load_prices(mk))
        if w:
            print(f"\n[캐시] {mk.upper()}: {w}")


# ───────────────────────────────────────────────────── 사후 분석 (--posthoc)

def _alt_score(P: dict, m: int, kind: str) -> np.ndarray:
    """MAX 대신 쓰는 월내 통계 — §6.8 플라시보."""
    sel = np.flatnonzero(P["codes"] == P["months"][m])
    seg = P["dret"][sel]
    with np.errstate(all="ignore"):
        if kind == "std":
            return np.nanstd(seg, axis=0)
        if kind == "min":
            return -np.nanmin(seg, axis=0)      # 부호 반전: 최악의 하루가 클수록 큰 값
        if kind == "mean":
            return np.nanmean(seg, axis=0)
        if kind == "resid":                      # MAX 를 월내 σ 에 횡단면 회귀한 잔차
            x = np.nanstd(seg, axis=0)
            y = P["max"][m]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                return np.full_like(y, np.nan)
            b = np.polyfit(x[ok], y[ok], 1)
            out = np.full_like(y, np.nan)
            out[ok] = y[ok] - (b[0] * x[ok] + b[1])
            return out
    raise ValueError(kind)


def _pit_mcap_ok(P: dict, m: int, mult: float) -> np.ndarray:
    """`mcap_t ≈ mcap_now × (close_t / close_now)` 로 과거 시총을 **근사**해
    당시에도 유니버스 문턱을 넘었을 종목만 남긴다 (§7-7).

    **편향을 제거하지 못한다** — 문턱 아래로 사라져 캐시에 아예 없는 종목은
    되살릴 수 없다. 방향과 대략의 크기만 본다."""
    import json
    from pipeline import config
    f = ("cache/universe_kr_stock.json" if P["market"] == "kr"
         else "cache/universe_us_stock_test.json")
    rows = json.load(open(f))
    cap = {r["ticker"]: r.get("market_cap_usd") for r in rows}
    tk = P["tickers"]
    now = P["close"][-1]
    thr = (config.KR_MIN_MCAP_KRW if P["market"] == "kr" else config.US_MIN_MCAP_USD) * mult
    px = P["pm"][m]
    ok = np.zeros(len(tk), dtype=bool)
    for j, t in enumerate(tk):
        c = cap.get(str(t))
        if c is None or not np.isfinite(now[j]) or now[j] <= 0 or not np.isfinite(px[j]):
            continue
        ok[j] = (c * px[j] / now[j]) >= thr
    return ok


def _spread_series(P: dict, *, kind=None, cap_ret=None, pit_mult=None,
                   net=True) -> pd.Series:
    cost = COST[P["market"]] if net else 0.0
    prev = [None] * N_DECILE
    vals, idx = [], []
    for m in range(1, len(P["me"]) - 1):
        sc = P["max"][m] if kind is None else _alt_score(P, m, kind)
        ok = eligible(P, m, score=sc)
        if pit_mult is not None:
            ok &= _pit_mcap_ok(P, m, pit_mult)
        decs = split_deciles(P, sc, ok)
        if not decs:
            continue
        drift = np.nan_to_num(P["mret"][m], nan=0.0)
        g, c = np.zeros(N_DECILE), np.zeros(N_DECILE)
        for d in range(N_DECILE):
            mem = decs[d]
            r = P["mret"][m + 1][mem]
            r = r[np.isfinite(r)]
            if cap_ret is not None:
                r = np.minimum(r, cap_ret)
            g[d] = float(r.mean()) * 100.0 if len(r) else 0.0
            cc, prev[d] = _turn_cost(prev[d], mem, drift, cost)
            c[d] = cc * 100.0
        vals.append((g[9] - g[0]) - (c[9] + c[0]))
        idx.append(P["months"][m])
    return pd.Series(vals, index=pd.PeriodIndex(idx, freq="M"), dtype=float)


def cmd_posthoc():
    """§6.8 플라시보 · §7-7 PIT 근사 시총 · §7-11 상한 · §7-13 레짐 · §7-14 결측.

    **전부 사후 분석이며 판정에 쓰지 않는다** (§3.0-4). 아카이브 재현용이다.
    """
    PC = _pc()
    P = _panels()

    def show(tag, **kw):
        ser = {m: _spread_series(P[m], **kw) for m in ("kr", "us")}
        drop = [m for m, v in ser.items() if len(v) < 12]
        for m in drop:                       # 필터가 너무 세서 표본이 남지 않는 시장
            del ser[m]
        if len(ser) < 2:
            print(f"  {tag:<34}  산출 불가 — 표본 부족 시장: {','.join(drop).upper()}")
            return None
        e = PC.all_estimates(ser)
        ts = [e["mu"] / e[k] for k in PC.VOTERS]
        rv, _, ag = PC.robust_verdict(e, K_CRIT, direction=LIT_DIR)
        print(f"  {tag:<34}{e['mu']:>+10.4f}%{min(ts):>+8.3f}{max(ts):>+8.3f}"
              f"  {'일치' if ag else '갈림':<4} {rv}"
              f"   n={e['n']}")
        return e

    print("=" * 104)
    print("[H-028 사후 분석] — **판정에 쓰지 않는다** (§3.0-4). 문서 §6.8·§7-7·§7-11·§7-13·§7-14 재현용")
    print("=" * 104)
    print(f"\n§6.8 플라시보 — MAX 대신 다른 월내 통계로 정렬 (**gross**, 비용 전)")
    print(f"  {'정렬 신호':<34}{'통합 월':>11}{'t 최소':>8}{'t 최대':>8}  판정")
    show("MAX (본안)", net=False)
    show("월내 표준편차", kind="std", net=False)
    show("월내 **최저** 일간수익 (반전)", kind="min", net=False)
    show("월내 평균 일간수익", kind="mean", net=False)
    show("MAX ⊥ σ (σ에 회귀한 잔차)", kind="resid", net=False)

    print(f"\n§7-11 이상치 — 종목-월 수익 상한 (**net**)")
    print(f"  {'변형':<34}{'통합 월':>11}{'t 최소':>8}{'t 최대':>8}  판정")
    base = show("본안 (상한 없음)")
    show("+100% 상한", cap_ret=1.0)
    p = np.concatenate([_spread_series(P[m]).to_numpy() for m in ("kr", "us")])
    print(f"    분포: 평균 {p.mean():+.4f} · 중위 {np.median(p):+.4f} · P(>0) {(p>0).mean()*100:.0f}%")

    print(f"\n§7-7 유니버스 전방참조 — PIT 근사 시총 필터 (**US 단독, net**)")
    print("  ⚠️ **KR 은 산출 불가** — `cache/universe_kr_stock.json` 의 `market_cap_usd`")
    print("     중위값이 5.99e8 인데 `config.KR_MIN_MCAP_KRW` 는 3.0e11 이라 단위가 맞지 않는다")
    print("     (문턱 통과 2/644). US 는 정합한다(3319/3319, 중위 3.56e9 vs 문턱 3.0e8).")
    print(f"  {'문턱 배수':<34}{'US 월':>11}{'t 최소':>8}{'t 최대':>8}  판정")
    for mult in (0.5, 0.8, 1.0, 1.5):
        ser = _spread_series(P["us"], pit_mult=mult)
        base_us = _spread_series(P["us"])
        t = tstat(ser.to_numpy())
        print(f"  ×{mult:<33}{ser.mean():>+10.4f}%{t:>+8.3f}{'':>8}"
              f"  naive t (기준 {base_us.mean():+.4f}%p / {tstat(base_us.to_numpy()):+.3f})")
    n0 = n1 = 0
    for m in range(1, len(P["us"]["me"]) - 1):
        ok = eligible(P["us"], m)
        n0 += int(ok.sum()); n1 += int((ok & _pit_mcap_ok(P["us"], m, 1.0)).sum())
    print(f"    US 적격 {n0/(len(P['us']['me'])-2):.1f} → {n1/(len(P['us']['me'])-2):.1f}종목"
          f" (탈락 {(1-n1/n0)*100:.1f}%)")

    print(f"\n§7-13 레짐 — 인라인 계산 vs btlib.regime.bull_map")
    from btlib import regime, loading as _l
    for mk in ("kr", "us"):
        bm = regime.bull_map(_l.load_bench(mk), sma=SMA_BENCH)
        dates = np.asarray(_l.load_prices(mk).pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last").sort_index().index)
        ref = np.array([bm.get(d, False) for d in dates])[P[mk]["me"]]
        print(f"    {mk.upper()} 불일치 {int((ref != P[mk]['bull']).sum())}/{len(ref)} (me 길이 기준)")

    print(f"\n§7-14 보유월 결측 비율 (D1 / D10)")
    for mk in ("kr", "us"):
        a = run_arm(P[mk])
        tot = [0, 0]; mis = [0, 0]
        for i, m in enumerate(a["months"]):
            ok = eligible(P[mk], m); decs = split_deciles(P[mk], P[mk]["max"][m], ok)
            for k, d in enumerate((0, N_DECILE - 1)):
                r = P[mk]["mret"][m + 1][decs[d]]
                tot[k] += len(r); mis[k] += int((~np.isfinite(r)).sum())
        print(f"    {mk.upper()} D1 {mis[0]/tot[0]*100:.2f}% · D10 {mis[1]/tot[1]*100:.2f}%")


# ────────────────────────────────────────────────────────────── selftest

def selftest():
    # ① MAX 정의 — 직전월 말 100, 그달 종가 [110, 99, 108]
    cl = np.array([[100.0], [110.0], [99.0], [108.0]])
    dr = np.full_like(cl, np.nan)
    dr[1:] = cl[1:] / cl[:-1] - 1.0
    got = np.nanmax(dr[1:], axis=0)[0]
    assert abs(got - 0.10) < 1e-12, got
    # ② 월 경계 — 직전월 말 종가를 바꾸면 MAX 가 바뀐다
    cl2 = cl.copy(); cl2[0] = 90.0
    dr2 = np.full_like(cl2, np.nan); dr2[1:] = cl2[1:] / cl2[:-1] - 1.0
    assert abs(np.nanmax(dr2[1:], axis=0)[0] - (110/90 - 1)) < 1e-12

    P0 = _synth(np.random.default_rng(0), n_m=40, n_tk=60)

    # ③ 최소 유효일 경계 — 14 부적격 / 15 적격
    Pm = dict(P0)
    Pm["ndays"] = P0["ndays"].copy()
    Pm["ndays"][5, :3] = [14, 15, 16]
    ok = eligible(Pm, 5)
    assert (not ok[0]) and ok[1] and ok[2], ok[:3]

    # ④ 십분위 분할
    Pd = {"tickers": np.array([f"t{i:04d}" for i in range(623)])}
    ds = split_deciles(Pd, np.arange(623.0), np.ones(623, bool))
    sz = [len(g) for g in ds]
    assert sum(sz) == 623 and max(sz) - min(sz) == 1 and sz[0] == 63, sz
    assert ds[0][0] == 0 and ds[9][-1] == 622
    assert set(ds[0]) & set(ds[9]) == set()
    assert split_deciles(Pd, np.arange(623.0), np.arange(623) < 19) == []

    # ⑤ 동점 → 티커 오름차순
    tk = np.array([f"z{19-i:02d}" for i in range(20)])
    dt = split_deciles({"tickers": tk}, np.zeros(20), np.ones(20, bool))
    assert list(tk[np.concatenate(dt)]) == sorted(tk)

    # ⑥ 비용 경계
    z = np.zeros(4)
    c1, w1 = _turn_cost(None, np.array([0, 1]), z, 1.0); assert abs(c1 - 1.0) < 1e-12
    c2, w2 = _turn_cost(w1, np.array([0, 1]), z, 1.0); assert abs(c2) < 1e-12
    c3, _ = _turn_cost(w2, np.array([2, 3]), z, 1.0); assert abs(c3 - 2.0) < 1e-12
    c4, _ = _turn_cost(w2, np.array([0, 1]), np.array([1.0, 0, 0, 0]), 1.0)
    assert abs(c4 - (abs(0.5 - 2/3) + abs(0.5 - 1/3))) < 1e-12

    # ⑦ 스프레드 부호 — 양 다리 비용 차감
    a = run_arm(P0)
    rhs = (a["gross"][:, 9] - a["gross"][:, 0]) - (a["cost"][:, 9] + a["cost"][:, 0])
    assert np.allclose(a["spread"], rhs, atol=1e-12)
    wrong = a["dec"][:, 9] - a["dec"][:, 0]
    assert np.allclose(wrong - a["spread"], 2.0 * a["cost"][:, 0], atol=1e-12)
    assert np.abs(wrong - a["spread"]).max() > 1e-6
    #    회전율은 두 다리 **평균** 편도
    want = (a["cost"][:, 9] + a["cost"][:, 0]) / COST[P0["market"]] / 100.0 / 4.0
    assert np.allclose(a["turn"], want, atol=1e-12)

    # ⑧ look-ahead — 뒤를 잘라도 월 m 의 MAX·십분위가 같다
    m0 = 20
    full_sc, full_ok = P0["max"][m0].copy(), eligible(P0, m0)
    full_d = [g.copy() for g in split_deciles(P0, full_sc, full_ok)]
    for cut in (m0 + 1, m0 + 3):
        Pc = dict(P0)
        for k in ("max", "ndays", "pm", "mret", "pit_dv", "cap_dv"):
            Pc[k] = P0[k][:cut]
        Pc["bull"] = P0["bull"][:cut]
        d2 = split_deciles(Pc, Pc["max"][m0], eligible(Pc, m0))
        assert all(np.array_equal(x, y) for x, y in zip(full_d, d2)), cut

    # ⑨ run_arm ↔ 루프 참조
    ref = _ref_arm(P0)
    assert np.allclose(a["gross"], ref["gross"], atol=1e-10)
    assert np.allclose(a["cost"], ref["cost"], atol=1e-10)
    assert np.allclose(a["spread"], ref["spread"], atol=1e-10)

    # ⑩ Bonferroni
    assert abs(K_CRIT - 2.2414) < 5e-5, K_CRIT

    # ⑪ 원문 EW 표의 순위상관 — 음수여야 한다
    rho_lit = spearman(np.arange(N_DECILE), LIT_EW_DECILES)
    assert rho_lit < 0, rho_lit
    #    스프레드도 원문 표에서 재현: 0.64 − 1.29 = −0.65
    assert abs((LIT_EW_DECILES[9] - LIT_EW_DECILES[0]) - LIT_EW_SPREAD) < 1e-9

    # ⑪-b **문헌 예측 부호 상수** — 이 가설의 채택/기각을 뒤집는 유일한 상수다
    assert LIT_DIR == int(np.sign(LIT_EW_SPREAD)) == -1, LIT_DIR
    assert DV_MIN["us"] == 2e6 and DV_MIN["kr"] is None      # ⑬(ㄹ) 자기 무력화 방지

    # ⑫ MAX(N) — N=1 이면 MAX 와 동일, N 이 커지면 값이 작아진다
    s1 = max_n(P0, m0, 1)
    assert np.allclose(s1, P0["max"][m0], equal_nan=True)
    s3 = max_n(P0, m0, 3)
    v = np.isfinite(s1) & np.isfinite(s3)
    assert (s3[v] <= s1[v] + 1e-12).all()
    assert (s3[v] < s1[v] - 1e-9).any()

    # ⑬ **build_panel 실데이터 경로** — 합성 패널이 우회하는 부분을 직접 덮는다
    #    (게이트가 변이 5종을 놓친 자리: MAX 정의·마스킹·PIT 거래대금·문턱)
    from btlib import loading as _ld
    for mkt, masked_expect in (("kr", 9008), ("us", None)):
        Q = build_panel(mkt)
        #    (ㅁ) 마지막 역월(부분 월)이 실제로 빠졌는가
        assert len(Q["months"]) == len(set(Q["codes"])) - 1, mkt
        raw = _ld.load_prices(mkt)
        pc = raw.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
        po = raw.pivot_table(index="date", columns="ticker", values="open", aggfunc="last").reindex(index=pc.index, columns=pc.columns)
        pv = raw.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").reindex(index=pc.index, columns=pc.columns)
        bad = ((pc <= 0) | (po <= 0)).to_numpy()
        if masked_expect is not None:                 # PROTOCOL §1 기재값
            assert int(bad.sum()) == masked_expect, (mkt, int(bad.sum()))
            assert int(bad.sum()) > 0
        #    (ㄱ) MAX 가 정말 **최댓값**인가 — 독립 재계산 (nanmin 이면 깨진다)
        m = len(Q["months"]) // 2
        sel = np.flatnonzero(Q["codes"] == Q["months"][m])
        c = np.where(bad, np.nan, pc.to_numpy(float))
        d = np.full_like(c, np.nan); d[1:] = c[1:] / c[:-1] - 1.0
        with np.errstate(all="ignore"):
            want = np.nanmax(d[sel], axis=0)
        assert np.allclose(Q["max"][m], want, equal_nan=True), mkt
        assert np.nanmax(Q["max"][m]) > 0, mkt        # 최댓값이면 양수가 존재한다
        #    (ㄴ) ndays 가 마스킹을 반영하는가
        assert np.array_equal(Q["ndays"][m], np.isfinite(d[sel]).sum(axis=0)), mkt
        assert Q["ndays"][m].max() <= len(sel)
        #    (ㄷ) PIT 거래대금이 **판정봉까지만** 본다 (look-ahead 없음)
        i = Q["me"][m]
        dvw = (np.where(bad, np.nan, (pc * pv).to_numpy(float)))[i - DV_WIN + 1:i + 1]
        with np.errstate(all="ignore"):
            wdv = dvw.mean(axis=0)                    # 결측 하나라도 있으면 NaN
        assert np.allclose(Q["pit_dv"][m], wdv, equal_nan=True), mkt
        #    (ㄹ) US 거래대금 문턱이 실제로 종목을 배제하는가
        if DV_MIN[mkt] is not None:
            ok_all = eligible(Q, m)
            n_thr = int((np.isfinite(Q["pit_dv"][m]) & (Q["pit_dv"][m] < DV_MIN[mkt])).sum())
            assert n_thr > 0 and not ok_all[np.flatnonzero(
                np.isfinite(Q["pit_dv"][m]) & (Q["pit_dv"][m] < DV_MIN[mkt]))].any()

    # ⑭ 가격 하한 해제가 실제로 유니버스를 넓히는가 (§3.2 민감도가 산출 가능한지)
    Qk = build_panel("kr")
    mid = len(Qk["months"]) // 2
    n_on = int(eligible(Qk, mid, use_floor=True).sum())
    n_off = int(eligible(Qk, mid, use_floor=False).sum())
    assert n_off > n_on, (n_on, n_off)
    assert run_arm(Qk, use_floor=False)["n"] > 0

    print("selftest: 14개 항목 통과 (MAX정의·월경계·최소유효일·십분위분할·동점·"
          "비용경계4종+회전율·스프레드부호·look-ahead·run_arm↔루프참조·Bonferroni·"
          "원문표ρ와스프레드·부호상수·MAX(N)·**build_panel실데이터**(MAX정의·마스킹·PIT·문턱·부분월)·"
          "**가격하한민감도**)")


def _synth(rng, n_m: int, n_tk: int) -> dict:
    """selftest 전용 합성 패널 — 역월 20영업일 고정."""
    D = 20
    n_d = n_m * D
    dates = pd.date_range("2015-01-01", periods=n_d, freq="B")
    dret = rng.normal(0.001, 0.02, size=(n_d, n_tk))
    dret[0] = np.nan
    cl = 100_000.0 * np.cumprod(1.0 + np.nan_to_num(dret, nan=0.0), axis=0)
    codes = pd.PeriodIndex([pd.Period(f"2015-01", "M") + (i // D) for i in range(n_d)],
                           freq="M")
    months = pd.PeriodIndex(sorted(set(codes)))[:n_m]
    me = np.array([np.flatnonzero(codes == p)[-1] for p in months])
    pm = cl[me]
    mret = np.full_like(pm, np.nan)
    mret[1:] = pm[1:] / pm[:-1] - 1.0
    mx = np.array([np.nanmax(dret[np.flatnonzero(codes == p)], axis=0) for p in months])
    nd = np.array([np.isfinite(dret[np.flatnonzero(codes == p)]).sum(axis=0)
                   for p in months], dtype=float)
    return {"tickers": np.array([f"s{i:03d}" for i in range(n_tk)]), "close": cl,
            "dret": dret, "me": me, "months": months, "n_days": n_d, "codes": codes,
            "pm": pm, "mret": mret, "max": mx, "ndays": nd,
            "pit_dv": np.full((n_m, n_tk), 9e9), "cap_dv": np.full((n_m, n_tk), 9e9),
            "bull": np.ones(n_m, bool), "market": "kr"}


def _ref_arm(P: dict) -> dict:
    """회계 부분만 딕셔너리 산술로 독립 재구현 (PROTOCOL §4-(b))."""
    cost = COST[P["market"]]
    prev = [None] * N_DECILE
    G, C = [], []
    for m in range(1, len(P["me"]) - 1):
        decs = split_deciles(P, P["max"][m], eligible(P, m))
        if not decs:
            continue
        g, c = [], []
        for d in range(N_DECILE):
            mem = list(map(int, decs[d]))
            w_new = {j: 1.0 / len(mem) for j in mem}
            if prev[d] is None:
                turn = sum(abs(v) for v in w_new.values())
            else:
                dr = {}
                for j, w in prev[d].items():
                    x = P["mret"][m][j]
                    dr[j] = w * (1.0 + (0.0 if not np.isfinite(x) else x))
                tot = sum(dr.values())
                dr = {j: v / tot for j, v in dr.items()} if tot > 0 else {}
                turn = sum(abs(w_new.get(j, 0.0) - dr.get(j, 0.0))
                           for j in set(w_new) | set(dr))
            vals = [P["mret"][m + 1][j] for j in mem]
            vals = [v for v in vals if np.isfinite(v)]
            g.append((sum(vals) / len(vals) if vals else 0.0) * 100.0)
            c.append(cost * turn * 100.0)
            prev[d] = w_new
        G.append(g); C.append(c)
    G, C = np.asarray(G), np.asarray(C)
    return {"gross": G, "cost": C,
            "spread": (G[:, 9] - G[:, 0]) - (C[:, 9] + C[:, 0])}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--power":
        cmd_power()
    elif arg == "--run":
        cmd_run()
    elif arg == "--posthoc":
        cmd_posthoc()
    else:
        print(__doc__)
