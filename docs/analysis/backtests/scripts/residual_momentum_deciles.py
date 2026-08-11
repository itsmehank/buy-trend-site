"""H-026 — 잔차 모멘텀 십분위. **Blitz·Huij·Martens (2011) 원문 형태 그대로.**

원문(JEF 18(3), pp.506-521)의 설계:
  · 회귀 (1): r_i,t = a_i + b·RMRF + b·SMB + b·HML + e_i,t  (월별 초과수익)
    → 이 데이터는 PIT 시총·장부가가 없어 **시장 1팩터**로 축소한다 (문서 §2.2 이탈 1).
  · 회귀 창 36개월 t-36..t-1. **36개월 이력이 완전한 종목만** 적격.
  · 형성 창 12-1M = t-12..t-2 (11개월).
  · 점수 = 형성 창 잔차 합 / 같은 기간 잔차 표준편차.
    **추정 알파를 넣지 않는다** → u = r - b̂·r_mkt (= e + â).
  · 점수 십분위 D10 - D1, **zero-investment**, 동일가중, 1개월 보유.
  · 레짐 필터 없음(원문에 없음).

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/residual_momentum_deciles.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/residual_momentum_deciles.py --power
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/residual_momentum_deciles.py --run

**--power 는 SE·검정력·비용만 출력한다. 평균·t값·부호는 출력 경로가 없다.**
사전등록 오염 방지 장치다 (H-015 §5.1과 동일).
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading

_N = NormalDist()

REG_MONTHS = 36                 # 회귀 창 t-36..t-1  →  월 M-35..M
FORM_LAG_FAR = 11               # 형성 창 시작   →  월 M-11
FORM_LAG_NEAR = 1               # 형성 창 끝     →  월 M-1
FORM_MONTHS = FORM_LAG_FAR - FORM_LAG_NEAR + 1        # 11
N_DECILE = 10
DV_WIN = 20                     # PROTOCOL §2 US 유동성 필터 창
DV_MIN = {"kr": None, "us": 2e6}
PRICE_FLOOR = {"kr": 1000.0, "us": 1.0}
COST = {"kr": 0.0014, "us": 0.0005}                   # 편도
SD_FLOOR = 1e-12       # std_W(u) > 0 의 수치 구현 (§2.4). 실측 SD 중앙값 ≈ 0.08
CAP_N = {"kr": 400, "us": 500}                        # 유니버스 캡 민감도용
CAP_WIN = 60
SMA_BENCH = 200

# 문헌 효과 (Table 2, 1M 보유, 연환산) — 사전 고정
LIT_ANN = {"resid": 11.20, "total": 10.26}
LIT_M = {"resid": LIT_ANN["resid"] / 12.0, "total": LIT_ANN["total"] / 12.0}
LIT_M_DIFF = (LIT_ANN["resid"] - LIT_ANN["total"]) / 12.0

FAMILY = 2                                             # §3.1 사전 고정
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414


# ────────────────────────────────────────────────────────────── 패널

def _cum(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN 안전 누적합과 유효 관측 누적수. 창 [a,b] 합 = C[b+1]-C[a]."""
    v = np.isfinite(a)
    b = np.where(v, a, 0.0)
    C = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(b, axis=0, out=C[1:])
    Nn = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(v.astype(float), axis=0, out=Nn[1:])
    return C, Nn


def month_end_positions(dates) -> np.ndarray:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return s.groupby([s.index.year, s.index.month]).last().to_numpy()


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)
    bad = (piv["close"] <= 0) | (piv["open"] <= 0)          # PROTOCOL §1
    for k in piv:
        piv[k] = piv[k].mask(bad)
    close = piv["close"]
    dv_d = (close * piv["volume"])
    pit_dv = dv_d.rolling(DV_WIN, min_periods=DV_WIN).mean().to_numpy(float)
    cap_dv = dv_d.rolling(CAP_WIN, min_periods=CAP_WIN).median().to_numpy(float)

    # 마지막 월말은 역월이 끝나기 전이라 **부분 월**이다 → 월 격자에서 제외 (§2.3)
    me = month_end_positions(idx)[:-1]
    cl = close.to_numpy(float)
    pm = cl[me]                                              # (n_m, n_tk) 월말 종가
    mret = np.full_like(pm, np.nan)
    mret[1:] = pm[1:] / pm[:-1] - 1.0

    b = loading.load_bench(market).set_index("date")["close"]
    b = b.reindex(pd.to_datetime(idx).date if not isinstance(idx[0], pd.Timestamp)
                  else idx).ffill()
    bv = b.to_numpy(float)
    bm = bv[me]
    mkt = np.full(len(me), np.nan)
    mkt[1:] = bm[1:] / bm[:-1] - 1.0
    sma = pd.Series(bv).rolling(SMA_BENCH, min_periods=SMA_BENCH).mean().to_numpy()

    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close": cl, "me": me, "n_days": len(idx),
            "mret": mret, "mkt": mkt, "pm": pm,
            "pit_dv": pit_dv[me], "cap_dv": cap_dv[me],
            "bull": (bm > sma[me]), "market": market}


# ────────────────────────────────────────────────────────── 점수·적격

def scores_at(P: dict, m: int, *, reg_months: int = REG_MONTHS,
              rf_m: float = 0.0, include_alpha: bool = False,
              standardize: bool = True) -> dict:
    """판정 월 m에서 잔차/총 모멘텀 점수와 적격 마스크를 낸다.

    회귀 창 R = [m-reg_months+1, m] (원문 t-36..t-1),
    형성 창 W = [m-11, m-1]        (원문 t-12..t-2).
    `include_alpha=True` 는 **원문이 배제한** 형태(평범한 OLS 잔차)로, 대조군이다.
    """
    a_r, b_r = m - reg_months + 1, m
    a_w, b_w = m - FORM_LAG_FAR, m - FORM_LAG_NEAR
    if a_r < 1:
        return {}
    Y = P["mret"][a_r:b_r + 1] - rf_m                    # (nR, ntk)
    x = P["mkt"][a_r:b_r + 1] - rf_m                     # (nR,)
    if not np.isfinite(x).all():
        return {}
    n = reg_months
    valid = np.isfinite(Y)
    nv = valid.sum(axis=0)
    Yf = np.where(valid, Y, 0.0)
    Sy = Yf.sum(axis=0)
    Sxy = (Yf * x[:, None]).sum(axis=0)
    Sx, Sxx = x.sum(), (x * x).sum()
    den = Sxx - Sx * Sx / n
    with np.errstate(all="ignore"):
        beta = (Sxy - Sx * Sy / n) / den
        alpha = Sy / n - beta * Sx / n

    # 형성 창 (R 안에 들어 있다)
    o0, o1 = a_w - a_r, b_w - a_r
    Yw, xw = Y[o0:o1 + 1], x[o0:o1 + 1]
    nw = Yw.shape[0]
    U = Yw - beta[None, :] * xw[:, None]
    if include_alpha:                                    # 원문이 배제한 형태
        U = U - alpha[None, :]
    su = U.sum(axis=0)
    with np.errstate(all="ignore"):
        sd = U.std(axis=0, ddof=1)
        resid = su / sd if standardize else su
        tot = np.exp(np.log1p(Yw + rf_m).sum(axis=0)) - 1.0

    ok = (nv == n) & np.isfinite(beta) & np.isfinite(resid) & np.isfinite(tot)
    if standardize:
        # 원문 조건은 std > 0 이지만, 시장을 완전히 추종하는 종목의 잔차는
        # 부동소수점상 정확히 0이 아니라 ~1e-18 이 된다(합산 순서 차이).
        # SD_FLOOR 는 그 수치 축퇴만 걸러낸다 — 실데이터 월별 잔차 SD 는 중앙값
        # KR 0.0906 · US 0.0777 이라 정상 종목을 배제하지 않는다(실측 배제 0건).
        # --run 이 배제 건수와 관측 최솟값을 출력한다.
        ok &= np.isfinite(sd) & (sd > SD_FLOOR)
    return {"resid": resid, "total": tot, "ok": ok, "beta": beta,
            "alpha": alpha, "nw": nw, "sd": sd, "nvalid": nv, "nreg": n}


def eligible(P: dict, m: int, base_ok: np.ndarray, *, cap: bool = False) -> np.ndarray:
    mk = P["market"]
    ok = base_ok.copy()
    px = P["pm"][m]
    ok &= np.isfinite(px) & (px >= PRICE_FLOOR[mk])          # 원문: price < $1 제외
    if DV_MIN[mk] is not None:                                # PROTOCOL §2 (US)
        dv = P["pit_dv"][m]
        ok &= np.isfinite(dv) & (dv >= DV_MIN[mk])
    if cap:                                                   # 민감도: 저장소 유니버스 캡
        dv = P["cap_dv"][m]
        ok &= np.isfinite(dv)
        cand = np.flatnonzero(ok)
        if len(cand) > CAP_N[mk]:
            thr = np.sort(dv[cand])[::-1][CAP_N[mk] - 1]
            ok &= dv >= thr
    return ok


def split_deciles(P: dict, score: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    """점수 **오름차순**(동점은 티커 오름차순) 10등분. [0]=D1(최저) … [9]=D10(최고)."""
    cand = np.flatnonzero(ok)
    if len(cand) < N_DECILE * 2:
        return []
    tk = P["tickers"]
    order = np.array(sorted(cand, key=lambda j: (score[j], tk[j])), dtype=int)
    return [np.asarray(g, dtype=int) for g in np.array_split(order, N_DECILE)]


# ─────────────────────────────────────────────────────────── 시뮬레이션

def _hold_return(P: dict, m: int, mem: np.ndarray, exec_lag: int) -> float:
    """월 m 종가에 잡은 바스켓의 **월 m+1** 동일가중 수익 (생존 종목 평균)."""
    if len(mem) == 0:
        return 0.0
    if exec_lag == 0:
        r = P["mret"][m + 1][mem]
    else:                                                     # T+1 종가 체결 민감도
        i0, i1 = P["me"][m] + exec_lag, P["me"][m + 1] + exec_lag
        if i1 >= P["n_days"]:
            return np.nan
        c = P["close"]
        r = c[i1][mem] / c[i0][mem] - 1.0
    r = r[np.isfinite(r)]
    return float(r.mean()) if len(r) else 0.0


def _turn_cost(prev: dict | None, mem: np.ndarray, drift_r: np.ndarray,
               cost: float) -> tuple[float, dict]:
    """c_oneway × Σ|w_new − w_drift|. 최초 진입은 Σ|Δw| = 1 (직전 비중 0)."""
    w_new = {int(j): 1.0 / len(mem) for j in mem} if len(mem) else {}
    if prev is None:
        drift = {}
    else:
        tmp = {}
        for j, w in prev.items():
            g = drift_r[j]
            tmp[j] = w * (1.0 + (g if np.isfinite(g) else 0.0))
        s = sum(tmp.values())
        drift = {j: v / s for j, v in tmp.items()} if s > 0 else {}
    keys = set(w_new) | set(drift)
    turn = sum(abs(w_new.get(j, 0.0) - drift.get(j, 0.0)) for j in keys)
    return cost * turn, w_new


def run_arm(P: dict, *, signal: str = "resid", reg_months: int = REG_MONTHS,
            rf_m: float = 0.0, include_alpha: bool = False,
            standardize: bool = True, cap: bool = False, exec_lag: int = 0,
            hold_k: int = 1) -> dict:
    """십분위 10칸의 월별 순수익(비용 후, %p)과 부속 통계.

    hold_k>1 은 원문의 중첩 포트폴리오 방식 — K개 코호트를 동시에 굴려 평균한다.
    """
    n_m = len(P["me"])
    cost = COST[P["market"]]
    start = max(reg_months, FORM_LAG_FAR + 1)
    rows, months = [], []
    prev = [[None] * N_DECILE for _ in range(hold_k)]
    uni_rows, bull_rows, ndec, elig_n = [], [], [], []
    cost_rows, turn_rows, gross_rows = [], [], []
    cohort_last = [None] * hold_k                      # 코호트별 직전 십분위 구성
    for m in range(start, n_m - 1):
        sc = scores_at(P, m, reg_months=reg_months, rf_m=rf_m,
                       include_alpha=include_alpha, standardize=standardize)
        if not sc:
            continue
        ok = eligible(P, m, sc["ok"], cap=cap)
        decs = split_deciles(P, sc[signal], ok)
        if not decs:
            continue
        c = m % hold_k                                  # 이번 달 재정렬할 코호트
        cohort_last[c] = decs
        if any(x is None for x in cohort_last):
            # 코호트가 다 채워지기 전에는 판정 표본에 넣지 않는다
            drift = np.nan_to_num(P["mret"][m], nan=0.0)
            for k in range(hold_k):
                if cohort_last[k] is None:
                    continue
                for d in range(N_DECILE):
                    _, prev[k][d] = _turn_cost(prev[k][d], cohort_last[k][d],
                                               drift, cost)
            continue
        drift = np.nan_to_num(P["mret"][m], nan=0.0)
        gro = np.zeros(N_DECILE)
        cst = np.zeros(N_DECILE)
        turn = 0.0
        for k in range(hold_k):
            mems = cohort_last[k]
            for d in range(N_DECILE):
                gross = _hold_return(P, m, mems[d], exec_lag)
                # gross 가 매월 동일가중 평균이므로 **모든 코호트가 매월 동일가중으로
                # 재조정**된다. 따라서 비용도 매월 부과한다 — 재정렬 달이 아닌
                # 코호트에 비용 0을 물리면 표류 복원을 공짜로 처리하게 된다.
                cc, prev[k][d] = _turn_cost(prev[k][d], mems[d], drift, cost)
                gro[d] += gross / hold_k
                cst[d] += cc / hold_k
                if d in (0, N_DECILE - 1) and cost > 0:
                    turn += cc / cost / hold_k / 2.0 / 2.0     # 두 다리의 **평균**
        net = gro - cst                       # 십분위를 **롱**으로 보유했을 때의 순수익
        uni = np.flatnonzero(ok)
        rows.append(net * 100.0)
        gross_rows.append(gro * 100.0)
        cost_rows.append(cst * 100.0)
        turn_rows.append(turn)
        uni_rows.append(_hold_return(P, m, uni, exec_lag) * 100.0)
        bull_rows.append(bool(P["bull"][m]))
        ndec.append(np.mean([len(g) for g in decs]))
        elig_n.append(len(uni))
        months.append(m)
    if not rows:
        return {}
    R, G, C = np.asarray(rows), np.asarray(gross_rows), np.asarray(cost_rows)
    # zero-investment 스프레드: 롱·숏 **양 다리 모두 비용을 차감**한다 (§2.5).
    #   spread = (g10 − g1) − (c10 + c1)
    # `R[:,9] − R[:,0]` 로 쓰면 D1 다리 비용이 **더해져** 버린다 (게이트 3회차 지적).
    spread = (G[:, N_DECILE - 1] - G[:, 0]) - (C[:, N_DECILE - 1] + C[:, 0])
    return {"dec": R, "gross": G, "spread": spread,
            "cost": np.asarray(cost_rows), "uni": np.asarray(uni_rows),
            "bull": np.asarray(bull_rows), "months": np.asarray(months),
            "turn": np.asarray(turn_rows),
            "ndec": float(np.mean(ndec)), "elig": float(np.mean(elig_n)),
            "n": len(rows)}


def cagr(monthly_pct: np.ndarray) -> float:
    """월 %p 계열의 연환산 기하수익. zero-investment 스프레드에도 관례적으로 쓴다."""
    g = np.prod(1.0 + np.asarray(monthly_pct, dtype=float) / 100.0)
    return float(g ** (12.0 / len(monthly_pct)) - 1.0) * 100.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """순위상관 (Spearman). §3.2가 요구하는 지표는 Pearson이 아니다."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# ────────────────────────────────────────────────────────────── 통계

def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def se_of(x: np.ndarray) -> float:
    """SE만. 평균·부호를 반환하지 않는다 (사전등록 오염 방지)."""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def power_at(se: float, effect: float, k: float = K_CRIT) -> float:
    d = effect / se
    return float(_N.cdf(d - k) + _N.cdf(-d - k))


def type_m(se: float, effect: float, k: float = K_CRIT) -> float:
    """E[|d̂| | 유의] / effect — 유의 시 과대추정 배율. **닫힌 형태**로 계산한다.

    `z ~ N(d, 1)` (d = effect/se) 일 때 유의 영역은 `z < −k` 와 `z > +k` 의
    **서로 떨어진 두 구간**이다. 이를 하나로 이어 사다리꼴 적분하면 `(−k, +k)` 를
    가로지르는 가짜 사다리꼴이 분자·분모에 더해진다 (게이트 3회차가 잡은 오류).

        P(|z| > k)          = Φ(d − k) + Φ(−d − k)
        E[ z · 1{z > k}]    =  d·Φ(d − k) + φ(k − d)
        E[−z · 1{z < −k}]   = −d·Φ(−d − k) + φ(k + d)
    """
    d = effect / se
    if d == 0:
        return float("nan")
    phi = lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    p = _N.cdf(d - k) + _N.cdf(-d - k)
    if p <= 0:
        return float("nan")
    num = (d * _N.cdf(d - k) + phi(k - d)) + (-d * _N.cdf(-d - k) + phi(k + d))
    return float((num / p) / d)


# ────────────────────────────────────────────────────────────── 명령

def _fmt(x, w=9, p=3, sign=True):
    return f"{x:>+{w}.{p}f}" if sign else f"{x:>{w}.{p}f}"


def cmd_power():
    print("=" * 100)
    print("[H-026] 사전 검출력 — **SE·검정력·비용만**. 평균·t·부호는 출력 경로가 없다.")
    print(f"  가족 {FAMILY}칸 · Bonferroni 양측 α=0.05 → |t| > {K_CRIT:.4f}")
    print(f"  문헌 효과: 잔차 D10−D1 = 월 {LIT_M['resid']:.4f}%p (연 {LIT_ANN['resid']}%)"
          f" · 잔차−총 = 월 {LIT_M_DIFF:.4f}%p")
    print("=" * 100)
    pooled_r, pooled_d = [], []
    for mk in ("kr", "us"):
        P = build_panel(mk)
        a = run_arm(P, signal="resid")
        b = run_arm(P, signal="total")
        n = min(a["n"], b["n"])
        sr, st = a["spread"][:n], b["spread"][:n]
        pooled_r.append(sr)
        pooled_d.append(sr - st)
        se = se_of(sr)
        print(f"\n── {mk.upper()} ── n={a['n']}개월 · 적격 평균 {a['elig']:.0f}종목"
              f" · 십분위 평균 {a['ndec']:.1f}종목")
        print(f"   D10−D1 SE = {se:.4f}%p · MDE(2.8016·SE) = {2.8016*se:.4f}%p"
              f" · 검정력 = {power_at(se, LIT_M['resid'])*100:.1f}%")
        cs = a["cost"]
        cm = float(cs[:, N_DECILE - 1].mean() + cs[:, 0].mean())
        print(f"   스프레드 월 비용 = {cm:.4f}%p (연 {cm*12:.2f}%p)"
              f" · 문헌 효과의 {cm*12/LIT_ANN['resid']*100:.0f}% 잠식")
    for tag, arr, eff in (("① 잔차 D10−D1", pooled_r, LIT_M["resid"]),
                          ("② 잔차−총 스프레드 차", pooled_d, LIT_M_DIFF)):
        p = np.concatenate(arr)
        se = se_of(p)
        pw = power_at(se, eff)
        print(f"\n[통합 {tag}] n={len(p)}개월 · SE = {se:.4f}%p"
              f" · MDE = {2.8016*se:.4f}%p")
        print(f"   문헌 효과 월 {eff:.4f}%p 기준 **검정력 = {pw*100:.1f}%**"
              f" · Type M = {type_m(se, eff):.2f}배")


def _verdict(t: float) -> str:
    if t > K_CRIT:
        return "**채택** (유의·양수 = 원문 방향)"
    if t < -K_CRIT:
        return "**기각** (유의·음수 = 원문 반대)"
    return "**측정 불가** (비유의)"


def cmd_run():
    print("=" * 100)
    print("[H-026] 잔차 모멘텀 십분위 — Blitz·Huij·Martens (2011) 원문 형태")
    print(f"  36개월 월별 1팩터 회귀 · 12−1M 표준화 · **알파 미포함** · D10−D1"
          f" zero-investment · 동일가중 · 레짐 없음")
    print(f"  판정: 통합 표본 · 가족 {FAMILY}칸 Bonferroni → |t| > {K_CRIT:.4f}")
    print("=" * 100)
    panels, arms = {}, {}
    pooled = {"resid": [], "total": [], "diff": [], "lo": []}
    for mk in ("kr", "us"):
        P = build_panel(mk)
        panels[mk] = P
        a = run_arm(P, signal="resid")
        b = run_arm(P, signal="total")
        arms[mk] = (a, b)
        n = min(a["n"], b["n"])
        sr, st = a["spread"][:n], b["spread"][:n]
        pooled["resid"].append(sr)
        pooled["total"].append(st)
        pooled["diff"].append(sr - st)
        lo = np.where(a["bull"][:n], a["dec"][:n, N_DECILE - 1] - a["uni"][:n], 0.0)
        pooled["lo"].append(lo)

        print(f"\n{'─'*100}\n── {mk.upper()} ──  n={a['n']}개월"
              f" · 적격 평균 {a['elig']:.0f}종목 · 십분위 {a['ndec']:.1f}종목"
              f" · 강세 {a['bull'].mean()*100:.0f}%")
        print(f"{'십분위':>7}{'잔차 월':>11}{'잔차 연':>10}{'총 월':>11}{'총 연':>10}")
        for d in range(N_DECILE):
            print(f"{'D'+str(d+1):>7}{_fmt(a['dec'][:,d].mean())}%"
                  f"{_fmt(a['dec'][:,d].mean()*12, 9, 2)}%"
                  f"{_fmt(b['dec'][:,d].mean())}%"
                  f"{_fmt(b['dec'][:,d].mean()*12, 9, 2)}%")
        for nm, s in (("잔차", sr), ("총수익", st)):
            ann = s.mean() * 12
            vol = s.std(ddof=1) * np.sqrt(12)
            print(f"  {nm} D10−D1  월 {s.mean():+.4f}%p · 연 {ann:+.2f}%p"
                  f" · t={tstat(s):+.2f} · 변동성 {vol:.2f}% · Sharpe {ann/vol:+.2f}"
                  f" · P(>0) {(s>0).mean()*100:.0f}% · CAGR {cagr(s):+.2f}%")
        rho_r = spearman(np.arange(N_DECILE), a["dec"].mean(axis=0))
        rho_t = spearman(np.arange(N_DECILE), b["dec"].mean(axis=0))
        print(f"  십분위 단조성 순위상관 ρ  잔차 {rho_r:+.3f} · 총수익 {rho_t:+.3f}")
        print(f"  회전율(D10·D1 평균, 편도 Σ|Δw|/2)  잔차 {a['turn'].mean()*100:.1f}%/월"
              f" · 총수익 {b['turn'].mean()*100:.1f}%/월")
        print(f"  D10 CAGR {cagr(a['dec'][:,N_DECILE-1]):+.2f}%"
              f" · D1 CAGR {cagr(a['dec'][:,0]):+.2f}%"
              f" · 유니버스 CAGR {cagr(a['uni']):+.2f}%")
        #    SD_FLOOR 가 실제로 몇 건을 배제했는지 (§2.4의 주장 검증).
        #    **다른 조건은 전부 통과했는데 문턱에만 걸린 건**만 센다.
        n_all = n_floor = 0
        sd_min = np.inf
        for mm in a["months"]:
            s = scores_at(P, int(mm))
            other = ((s["nvalid"] == s["nreg"]) & np.isfinite(s["beta"])
                     & np.isfinite(s["total"]) & np.isfinite(s["sd"]))
            n_all += int(other.sum())
            n_floor += int((other & (s["sd"] <= SD_FLOOR)).sum())
            if other.any():
                sd_min = min(sd_min, float(np.nanmin(s["sd"][other])))
        print(f"  SD_FLOOR(1e-12) 배제: **{n_floor}건** / 다른 조건 통과 {n_all:,} 종목-월"
              f" · 관측된 std_W(u) 최솟값 {sd_min:.3e}")
        se_m = se_of(sr)
        print(f"  [참고] 잔차 D10−D1 95% CI = "
              f"[{sr.mean()-1.96*se_m:+.4f}, {sr.mean()+1.96*se_m:+.4f}] %p/월"
              f"  · 문헌값 {LIT_M['resid']:.4f} 포함? "
              f"{'예' if abs(sr.mean()-LIT_M['resid']) < 1.96*se_m else '아니오'}")

    print(f"\n{'='*100}\n[통합 표본 판정]")
    res = {}
    for tag, key, eff in (("① 잔차 D10−D1", "resid", LIT_M["resid"]),
                          ("② 잔차−총 차", "diff", LIT_M_DIFF)):
        p = np.concatenate(pooled[key])
        t = tstat(p)
        res[key] = (p, t)
        se = se_of(p)
        lo, hi = p.mean() - 1.96 * se, p.mean() + 1.96 * se
        print(f"\n{tag}: n={len(p)}개월 · 월 {p.mean():+.4f}%p"
              f" · 연 {p.mean()*12:+.2f}%p · **t = {t:+.3f}**"
              f"  (문헌 예측 월 {eff:+.4f}%p)")
        print(f"   95% CI = [{lo:+.4f}, {hi:+.4f}] %p/월"
              f" = [{lo*12:+.2f}, {hi*12:+.2f}] %p/년")
        print(f"   0 포함? {'예' if lo <= 0 <= hi else '아니오'}"
              f"  ·  문헌값({eff:.4f}) 포함? {'예' if lo <= eff <= hi else '아니오'}")
        print(f"   → {_verdict(t)}")
        if abs(t) > K_CRIT:
            print(f"   Type M 배율 = {type_m(se, eff):.2f}배 (유의 시 과대추정 정도)")
    pt = np.concatenate(pooled["total"])
    print(f"\n[병기] 총수익 D10−D1 통합: 월 {pt.mean():+.4f}%p · t = {tstat(pt):+.3f}")
    plo = np.concatenate(pooled["lo"])
    print(f"[병기·**오염**] 롱온리 D10−유니버스(레짐 적용): 월 {plo.mean():+.4f}%p"
          f" · t = {tstat(plo):+.3f}   ← 문서 §7-1 참조")

    print(f"\n[가족 크기 민감도 — 판정에 쓰지 않는다]")
    print(f"{'가족':>5}{'임계|t|':>10}{'① 판정':>28}")
    for f in (1, 2, 4, 12):
        k = _N.inv_cdf(1.0 - 0.05 / (2.0 * f))
        t = res["resid"][1]
        v = "채택" if t > k else ("기각" if t < -k else "측정 불가")
        star = "  ← 사전 고정" if f == FAMILY else ""
        print(f"{f:>5}{k:>10.4f}{v:>20}{star}")

    print(f"\n{'='*100}\n[견고성 — 전부 병기, 판정에 쓰지 않는다]")
    variants = [
        ("알파 **포함** (원문이 배제한 형태)", dict(include_alpha=True)),
        ("비표준화 (원문 각주 2)", dict(standardize=False)),
        ("회귀창 60개월 (원문 §5)", dict(reg_months=60)),
        ("rf = 연 2% 상수", dict(rf_m=0.02 / 12.0)),
        ("T+1 종가 체결", dict(exec_lag=1)),
        (f"유니버스 캡 {CAP_N['kr']}/{CAP_N['us']}", dict(cap=True)),
        ("보유 3개월 (중첩)", dict(hold_k=3)),
        ("보유 6개월 (중첩)", dict(hold_k=6)),
        ("보유 12개월 (중첩)", dict(hold_k=12)),
    ]
    print(f"{'변형':<34}{'통합 월':>11}{'연':>9}{'t':>9}{'n':>6}")
    for name, kw in variants:
        segs = []
        for mk in ("kr", "us"):
            r = run_arm(panels[mk], signal="resid", **kw)
            if r:
                segs.append(r["spread"])
        if not segs:
            continue
        p = np.concatenate(segs)
        print(f"{name:<34}{p.mean():>+10.4f}%{p.mean()*12:>+8.2f}%"
              f"{tstat(p):>+9.2f}{len(p):>6}")

    for mk in ("kr", "us"):
        w = loading.staleness_warning(loading.load_prices(mk))
        if w:
            print(f"\n[캐시] {mk.upper()}: {w}")


# ────────────────────────────────────────────────────────────── selftest

def selftest():
    # ① 차원 — Σu / std(u) 는 무차원. 스케일을 c배 하면 점수가 불변이어야 한다.
    rng = np.random.default_rng(0)
    u = rng.normal(size=11)
    s1 = u.sum() / u.std(ddof=1)
    s2 = (3.7 * u).sum() / (3.7 * u).std(ddof=1)
    assert abs(s1 - s2) < 1e-10, (s1, s2)

    # ② β 소형 예제 — r_mkt=[1,2,3]%, r_i=[2,4,6]% → b̂=2.0, â=0
    x = np.array([0.01, 0.02, 0.03])
    y = np.array([0.02, 0.04, 0.06])
    n = 3
    b = ((x * y).sum() - x.sum() * y.sum() / n) / ((x * x).sum() - x.sum() ** 2 / n)
    a = y.mean() - b * x.mean()
    assert abs(b - 2.0) < 1e-12 and abs(a) < 1e-15, (b, a)
    #    비퇴화 예제 (â ≠ 0) — y = 2x + 0.01 이면 b̂=2.0, â=0.01, e≡0
    y2 = 2 * x + 0.01
    b2 = ((x * y2).sum() - x.sum() * y2.sum() / n) / ((x * x).sum() - x.sum() ** 2 / n)
    a2 = y2.mean() - b2 * x.mean()
    assert abs(b2 - 2.0) < 1e-12 and abs(a2 - 0.01) < 1e-15, (b2, a2)
    #    비퇴화 + 잔차 있는 예제
    y3 = np.array([0.03, 0.05, 0.10])
    b3 = ((x * y3).sum() - x.sum() * y3.sum() / n) / ((x * x).sum() - x.sum() ** 2 / n)
    a3 = y3.mean() - b3 * x.mean()
    e3 = y3 - a3 - b3 * x
    assert abs(b3 - 3.5) < 1e-12, b3
    assert abs(a3 - (0.06 - 3.5 * 0.02)) < 1e-15, a3
    assert abs(e3.sum()) < 1e-15 and abs((e3 * x).sum()) < 1e-17   # OLS 직교

    # ③ 알파 항등식 u = e + â
    u3 = y3 - b3 * x
    assert np.allclose(u3, e3 + a3, atol=1e-15)

    # ④ 표준화 불변 — std(u) = std(e)
    assert abs(u3.std(ddof=1) - e3.std(ddof=1)) < 1e-15

    # ⑤ 알파 포함/미포함의 **방향** — 원문 인용 6의 논거를 수치로 재현.
    #    t-36..t-13 에만 큰 양수 수익이 있고 형성창 W 는 0 인 종목 → â > 0 이므로
    #    알파를 **빼면**(=원문이 말하는 include) 점수가 낮아진다("would rank low").
    nR = 36
    xr = np.zeros(nR)
    yr = np.zeros(nR)
    yr[:24] = 0.05                      # t-36..t-13 구간(앞 24개월)만 큰 양수
    br = 0.0
    ar = yr.mean()
    assert ar > 0
    W = slice(nR - 11 - 1, nR - 1)      # 형성창 [m-11, m-1]
    uw = yr[W] - br * xr[W]
    ew = uw - ar
    assert uw.sum() > ew.sum(), (uw.sum(), ew.sum())
    assert abs(uw.sum()) < 1e-15        # W 구간 수익 0 → 알파 미포함 점수 분자 0
    assert ew.sum() < 0                 # 알파 포함이면 음수 → rank low ✓

    # ⑥ 극단: 완전 시장 추종 r_i = 2·r_mkt → std_W(u)=0 → **scores_at 이 실제로 배제**
    nM, m0 = 60, 50
    mk6 = rng.normal(0.008, 0.04, size=nM)
    mk6[0] = np.nan
    r6 = rng.normal(0.01, 0.06, size=(nM, 3))
    r6[0] = np.nan
    r6[1:, 1] = 2.0 * mk6[1:]                 # 종목 1: 완전 시장 추종
    s6 = scores_at({"mret": r6, "mkt": mk6}, m0)
    assert s6["ok"][0] and s6["ok"][2], s6["ok"]
    assert not s6["ok"][1], "완전 시장 추종 종목이 배제되지 않았다"
    assert abs(s6["beta"][1] - 2.0) < 1e-9, s6["beta"][1]
    #    정상 종목의 잔차 SD 는 SD_FLOOR 보다 8자리 이상 크다 (문턱이 무해함을 확인)
    Y6 = r6[m0 - 11:m0]
    sd6 = (Y6[:, 0] - s6["beta"][0] * mk6[m0 - 11:m0]).std(ddof=1)
    assert sd6 > 1e4 * SD_FLOOR, sd6

    # ⑦ 십분위 분할 — 623종목
    g = np.array_split(np.arange(623), 10)
    sizes = [len(z) for z in g]
    assert sum(sizes) == 623 and max(sizes) - min(sizes) <= 1, sizes
    assert len(set(g[0]) & set(g[9])) == 0

    # ⑧ 비용 경계 — 완전 교체 Σ|Δw|=2, 무변동 0, 최초 진입 1
    dr = np.zeros(4)
    c, w1 = _turn_cost(None, np.array([0, 1]), dr, 1.0)
    assert abs(c - 1.0) < 1e-12, c                      # 최초 진입
    c2, w2 = _turn_cost(w1, np.array([0, 1]), dr, 1.0)
    assert abs(c2) < 1e-12, c2                          # 무변동
    c3, _ = _turn_cost(w2, np.array([2, 3]), dr, 1.0)
    assert abs(c3 - 2.0) < 1e-12, c3                    # 완전 교체
    #    표류 반영 — 한 종목만 +100% 오르면 비중이 2/3:1/3 로 벌어진다
    dr2 = np.array([1.0, 0.0, 0.0, 0.0])
    c4, _ = _turn_cost(w2, np.array([0, 1]), dr2, 1.0)
    assert abs(c4 - (abs(0.5 - 2/3) + abs(0.5 - 1/3))) < 1e-12, c4
    #    run_arm 의 `turn` 은 **편도**(Σ|Δw|/2)의 **두 다리 평균**이어야 한다
    Pt8 = _synth_panel(np.random.default_rng(11), n_m=70, n_tk=60, stable=False)
    a8 = run_arm(Pt8, signal="resid")
    want = ((a8["cost"][:, N_DECILE - 1] + a8["cost"][:, 0])
            / COST[Pt8["market"]] / 100.0 / 2.0 / 2.0)
    assert np.allclose(a8["turn"], want, atol=1e-12), \
        float(np.abs(a8["turn"] - want).max())
    assert 0.0 < a8["turn"].mean() < 1.0, a8["turn"].mean()   # 편도 회전율의 범위

    # ⑨ look-ahead — 판정 월 m 의 점수는 m 이후 데이터를 지워도 같다
    n_m, n_tk = 60, 8
    r = rng.normal(0.01, 0.06, size=(n_m, n_tk))
    r[0] = np.nan
    mkt = rng.normal(0.008, 0.04, size=n_m)
    mkt[0] = np.nan
    Pfull = {"mret": r, "mkt": mkt}
    m0 = 50
    full = scores_at(Pfull, m0)
    for cut in (m0 + 1, m0 + 3):
        Pcut = {"mret": r[:cut].copy(), "mkt": mkt[:cut].copy()}
        got = scores_at(Pcut, m0)
        assert np.allclose(full["resid"], got["resid"], equal_nan=True), cut
        assert np.allclose(full["total"], got["total"], equal_nan=True), cut

    # ⑩ 월수익 산식 — 2종목 3개월 소형 예제, 결측 1건
    pm = np.array([[100.0, 50.0], [110.0, np.nan], [121.0, 55.0]])
    mr = np.full_like(pm, np.nan)
    mr[1:] = pm[1:] / pm[:-1] - 1.0
    assert np.allclose(mr[1, 0], 0.10) and np.isnan(mr[1, 1])
    assert np.allclose(mr[2, 0], 0.10) and np.isnan(mr[2, 1])

    # ⑪ Bonferroni 임계
    assert abs(K_CRIT - 2.2414) < 5e-5, K_CRIT
    for f, k in ((1, 1.9600), (4, 2.4977), (12, 2.8653)):
        assert abs(_N.inv_cdf(1 - 0.05 / (2 * f)) - k) < 5e-5, (f, k)

    # ⑫ 점수의 횡단면 표준편차 — Σu/s = √11 · t₁₀ 이므로 sd = √11·√(10/8) = 3.708
    pred = np.sqrt(FORM_MONTHS) * np.sqrt(10 / 8)
    assert abs(pred - 3.7081) < 1e-3, pred
    g = rng.normal(0.0, 0.08, size=(FORM_MONTHS, 200000))
    sc = g.sum(axis=0) / g.std(axis=0, ddof=1)
    assert abs(sc.std(ddof=1) / pred - 1.0) < 0.02, (sc.std(ddof=1), pred)

    # ⑬ 형성창이 [m−11, m−1] 인가 — 월 m 은 **회귀(β)를 통해서만** 점수에 들어간다.
    #    월 m 을 흔들어 β 를 바꾼 뒤, 점수가 "형성창 수익(불변) + 새 β" 로 정확히
    #    재구성되는지 본다. 형성창이 월 m 을 포함했다면(b_w = m) 이 재구성이 깨진다.
    mkt13 = rng.normal(0.008, 0.04, size=60)
    mkt13[0] = np.nan
    r13 = rng.normal(0.01, 0.06, size=(60, 4))
    r13[0] = np.nan
    rA = r13.copy()
    rA[m0] += 0.5                                   # 형성창 **밖**(최근 1개월)을 흔든다
    sA = scores_at({"mret": rA, "mkt": mkt13}, m0)
    bA = sA["beta"]
    #    형성창 수익은 **원본 배열**에서 가져온다 — 월 m 만 흔들었으므로 동일해야 한다
    Yw13 = r13[m0 - FORM_LAG_FAR:m0 - FORM_LAG_NEAR + 1]
    xw13 = mkt13[m0 - FORM_LAG_FAR:m0 - FORM_LAG_NEAR + 1]
    assert Yw13.shape[0] == FORM_MONTHS == 11, Yw13.shape
    U13 = Yw13 - bA[None, :] * xw13[:, None]
    pred = U13.sum(axis=0) / U13.std(axis=0, ddof=1)
    assert np.allclose(sA["resid"], pred, atol=1e-12), \
        f"형성창이 [m−11, m−1] 이 아니다 (최대 {np.abs(sA['resid'] - pred).max():.3g})"
    #    β 가 실제로 움직였는지 확인 — 안 움직였다면 위 검증이 공허하다
    b0 = scores_at({"mret": r13, "mkt": mkt13}, m0)["beta"]
    assert np.abs(bA - b0).max() > 1e-3, np.abs(bA - b0).max()

    # ⑭ split_deciles — 크기·D1/D10 배치·동점 티커 처리
    Pd = {"tickers": np.array([f"t{i:04d}" for i in range(623)])}
    sc14 = np.arange(623, dtype=float)                    # 점수 = 인덱스 (동점 없음)
    ds = split_deciles(Pd, sc14, np.ones(623, dtype=bool))
    sz = [len(g) for g in ds]
    assert sum(sz) == 623 and max(sz) - min(sz) == 1, sz
    assert sz[0] == 63, sz                                # 앞 조각이 크다 → D1 이 63
    assert ds[0][0] == 0 and ds[N_DECILE - 1][-1] == 622  # 오름차순 → [9]=D10(최고)
    assert set(ds[0]) & set(ds[N_DECILE - 1]) == set()
    #    동점은 티커 오름차순 — **split_deciles 의 반환 순서로** 확인한다.
    #    20종목 전부 점수 동일, 티커는 역순으로 준다 → 결과는 티커 오름차순이어야 한다
    tk20 = np.array([f"z{19-i:02d}" for i in range(20)])
    Pt = {"tickers": tk20}
    dt = split_deciles(Pt, np.zeros(20), np.ones(20, dtype=bool))
    assert [len(g) for g in dt] == [2] * 10, [len(g) for g in dt]
    flat = np.concatenate(dt)
    assert list(tk20[flat]) == sorted(tk20), list(tk20[flat])
    #    적격 미달(2·N_DECILE 미만)이면 빈 리스트
    assert split_deciles(Pd, sc14, np.arange(623) < 19) == []

    # ⑮ eligible — 가격 하한·거래대금 필터가 실제로 배제하는가
    Pe = {"market": "us", "tickers": np.array(["a", "b", "c", "d"]),
          "pm": np.array([[10.0, 0.5, 20.0, 30.0]]),           # b 는 $1 미만
          "pit_dv": np.array([[5e6, 5e6, 1e6, np.nan]]),       # c 미달, d 결측
          "cap_dv": np.array([[9e6, 9e6, 9e6, 9e6]])}
    got = eligible(Pe, 0, np.ones(4, dtype=bool))
    assert list(got) == [True, False, False, False], got
    Pk = dict(Pe, market="kr", pm=np.array([[2000.0, 500.0, 2000.0, 2000.0]]))
    got = eligible(Pk, 0, np.ones(4, dtype=bool))
    assert list(got) == [True, False, True, True], got     # KR 은 거래대금 필터 없음

    # ⑯ run_arm — **소표본 루프 참조 구현과 대조** (PROTOCOL §4-(b))
    P16 = _synth_panel(rng, n_m=70, n_tk=60, stable=False)
    got = run_arm(P16, signal="resid")
    ref = _ref_arm(P16, signal="resid")
    assert got["n"] == ref["n"] and got["n"] > 20, (got["n"], ref["n"])
    assert np.allclose(got["dec"], ref["dec"], atol=1e-10), \
        float(np.abs(got["dec"] - ref["dec"]).max())
    assert np.allclose(got["cost"], ref["cost"], atol=1e-10)

    # ⑰ hold_k 코호트 회계 — 십분위 구성이 시간불변이면 hold_k=K ≡ hold_k=1
    #    (마지막 코호트가 첫 진입비용을 무는 **첫 공통월 1개만** 전이 구간이라 제외한다.
    #     그 한 달을 빼지 않으면 아래 단언은 실제로 깨진다 — 형식적 검증이 아니다.)
    P17 = _synth_panel(rng, n_m=90, n_tk=60, stable=True)
    a1 = run_arm(P17, signal="resid", hold_k=1)
    #    전제 확인: 구성이 정말 시간불변인가
    seen = set()
    for m in range(REG_MONTHS, len(P17["me"]) - 1):
        s = scores_at(P17, m)
        ds = split_deciles(P17, s["resid"], eligible(P17, m, s["ok"]))
        if ds:
            seen.add(tuple(tuple(int(j) for j in g) for g in ds))
    assert len(seen) == 1, f"합성 패널의 십분위 구성이 시간불변이 아니다 ({len(seen)}종)"
    for K in (3, 6):
        aK = run_arm(P17, signal="resid", hold_k=K)
        common = np.intersect1d(a1["months"], aK["months"])[1:]      # 전이 1개월 제외
        assert len(common) > 20, len(common)
        i1 = np.searchsorted(a1["months"], common)
        iK = np.searchsorted(aK["months"], common)
        assert np.allclose(a1["spread"][i1], aK["spread"][iK], atol=1e-10), \
            (K, float(np.abs(a1["spread"][i1] - aK["spread"][iK]).max()))
        assert np.allclose(a1["cost"][i1], aK["cost"][iK], atol=1e-10), K

    # ⑱ type_m — 닫힌 형태를 **몬테카를로**와 대조 (게이트 3회차가 잡은 적분 오류)
    for d_true, se in ((0.9333, 0.3469), (0.0783, 0.1915)):
        got = type_m(se, d_true)
        d = d_true / se
        z = rng.normal(d, 1.0, size=4_000_000)
        sel = np.abs(z) > K_CRIT
        mc = float(np.abs(z[sel]).mean() / d)
        assert abs(got / mc - 1.0) < 0.01, (d_true, got, mc)
        assert got >= 1.0 - 1e-9, got            # 절단 평균은 참값 이상이어야 한다
    #    d → ∞ 면 배율 → 1 (거의 항상 유의하므로 선택 편향이 사라진다)
    assert abs(type_m(0.01, 1.0) - 1.0) < 1e-6, type_m(0.01, 1.0)

    # ⑲ spearman / cagr — 손계산 대조
    assert abs(spearman(np.arange(5), np.array([1.0, 2, 3, 4, 5])) - 1.0) < 1e-12
    assert abs(spearman(np.arange(5), np.array([5.0, 4, 3, 2, 1])) + 1.0) < 1e-12
    #    Spearman 은 단조변환에 불변, Pearson 은 아니다 → 둘이 실제로 다름을 확인
    y19 = np.array([1.0, 2, 3, 4, 100])
    assert abs(spearman(np.arange(5), y19) - 1.0) < 1e-12
    assert abs(np.corrcoef(np.arange(5), y19)[0, 1] - 1.0) > 0.1
    #    cagr: 월 1% 12개월 → (1.01^12 − 1) = 12.6825%
    assert abs(cagr(np.full(12, 1.0)) - (1.01 ** 12 - 1) * 100) < 1e-9
    assert abs(cagr(np.zeros(24))) < 1e-12

    # ⑳ 스프레드 비용 부호 — 양 다리 비용이 **차감**되는가
    P20 = _synth_panel(rng, n_m=70, n_tk=60, stable=False)
    a20 = run_arm(P20, signal="resid")
    lhs = a20["spread"]
    rhs = ((a20["gross"][:, N_DECILE - 1] - a20["gross"][:, 0])
           - (a20["cost"][:, N_DECILE - 1] + a20["cost"][:, 0]))
    assert np.allclose(lhs, rhs, atol=1e-12)
    #    net 차분(= 잘못된 형태)과는 실제로 달라야 한다 — D1 비용이 두 번 들어간다
    wrong = a20["dec"][:, N_DECILE - 1] - a20["dec"][:, 0]
    assert np.allclose(wrong - lhs, 2.0 * a20["cost"][:, 0], atol=1e-12)
    assert np.abs(wrong - lhs).max() > 1e-6, "비용이 0이라 이 검증이 무의미하다"

    print("selftest: 20개 항목 통과 (차원·β소형3종·알파항등식·표준화불변·알파방향·"
          "완전추종배제·십분위분할·비용경계4종·look-ahead·월수익·Bonferroni·"
          "점수산포3.708·형성창스킵(실질)·split_deciles·eligible·run_arm↔루프참조·"
          "hold_k코호트회계·type_m↔MC·spearman/cagr·스프레드비용부호)")


def _synth_panel(rng, n_m: int, n_tk: int, stable: bool) -> dict:
    """selftest 전용 합성 패널.

    stable=True 는 **십분위 구성이 시간불변**이 되도록 만든다 —
    잡음을 주기 11의 사인파로 주면 임의의 11개월 창에서 합이 0이고 표준편차가
    같으므로, 점수 = (11·drift_i)/(eps·s₀) 로 drift 순서에 의해서만 결정된다.
    (잡음을 아예 0으로 두면 잔차 SD 가 SD_FLOOR 아래로 내려가 전 종목이 배제된다.)
    """
    mkt = rng.normal(0.008, 0.04, size=n_m)
    mkt[0] = np.nan
    beta = rng.uniform(0.5, 1.5, size=n_tk)
    drift = np.linspace(-0.02, 0.02, n_tk)
    r = beta[None, :] * mkt[:, None] + drift[None, :]
    if stable:
        wave = 1e-3 * np.sin(2 * np.pi * np.arange(n_m) / FORM_MONTHS)
        r = r + wave[:, None]                      # 전 종목 동일 → 분모가 같아진다
    else:
        r = r + rng.normal(0.0, 0.05, size=(n_m, n_tk))
    r[0] = np.nan
    base = 100_000.0                 # KR 가격 하한(1,000원)을 넉넉히 넘기게 잡는다
    pm = np.vstack([np.full(n_tk, base),
                    base * np.cumprod(1.0 + np.nan_to_num(r[1:], nan=0.0), axis=0)])
    return {"market": "kr", "tickers": np.array([f"s{i:03d}" for i in range(n_tk)]),
            "mret": r, "mkt": mkt, "pm": pm,
            "pit_dv": np.full((n_m, n_tk), 9e9), "cap_dv": np.full((n_m, n_tk), 9e9),
            "bull": np.ones(n_m, dtype=bool), "me": np.arange(n_m),
            "close": pm, "n_days": n_m}


def _ref_arm(P: dict, signal: str = "resid") -> dict:
    """run_arm 의 **회계 부분만** 딕셔너리 산술로 다시 구현한 참조.

    점수·적격·십분위는 같은 함수를 쓰고(그쪽은 ⑭⑮가 따로 검증),
    보유수익·표류·비용 누적을 독립 경로로 계산해 벡터화 구현과 대조한다.
    """
    n_m = len(P["me"])
    cost = COST[P["market"]]
    start = max(REG_MONTHS, FORM_LAG_FAR + 1)
    prev = [None] * N_DECILE
    dec_rows, cost_rows, gro_rows = [], [], []
    for m in range(start, n_m - 1):
        sc = scores_at(P, m)
        if not sc:
            continue
        ok = eligible(P, m, sc["ok"])
        decs = split_deciles(P, sc[signal], ok)
        if not decs:
            continue
        net, cst, gro = [], [], []
        for d in range(N_DECILE):
            mem = list(map(int, decs[d]))
            w_new = {j: 1.0 / len(mem) for j in mem}
            if prev[d] is None:
                turn = sum(abs(v) for v in w_new.values())
            else:
                dr = {}
                for j, w in prev[d].items():
                    g = P["mret"][m][j]
                    dr[j] = w * (1.0 + (0.0 if not np.isfinite(g) else g))
                tot = sum(dr.values())
                dr = {j: v / tot for j, v in dr.items()} if tot > 0 else {}
                turn = sum(abs(w_new.get(j, 0.0) - dr.get(j, 0.0))
                           for j in set(w_new) | set(dr))
            cc = cost * turn
            vals = [P["mret"][m + 1][j] for j in mem]
            vals = [v for v in vals if np.isfinite(v)]
            gross = sum(vals) / len(vals) if vals else 0.0
            net.append((gross - cc) * 100.0)
            cst.append(cc * 100.0)
            gro.append(gross * 100.0)
            prev[d] = w_new
        dec_rows.append(net)
        cost_rows.append(cst)
        gro_rows.append(gro)
    D, C, G = (np.asarray(dec_rows), np.asarray(cost_rows), np.asarray(gro_rows))
    return {"dec": D, "cost": C, "gross": G, "n": len(dec_rows),
            "spread": (G[:, N_DECILE - 1] - G[:, 0]) - (C[:, N_DECILE - 1] + C[:, 0])}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--power":
        cmd_power()
    elif arg == "--run":
        cmd_run()
    else:
        print(__doc__)
