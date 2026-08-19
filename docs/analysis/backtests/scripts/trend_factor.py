"""H-033 — 트렌드 팩터 (H-029 재시도). **Han·Zhou·Zhu (2016, JFE 122(2) 352-375) 원문 형태 그대로.**

원문 식 (1)(2)(3)(4)(5):

    A_{j,t,L} = (P_{j,d-L+1} + … + P_{j,d}) / L          … (1)  d = 월 t 마지막 거래일
    Ã_{j,t,L} = A_{j,t,L} / P_{j,d}                       … (2)  종가로 정규화
    r_{j,t}   = β_{0,t} + Σ_i β_{i,t} Ã_{j,t-1,L_i} + ε   … (3)  **월별 횡단면 회귀**
    E_t[r_{j,t+1}] = Σ_i E_t[β_{i,t+1}] Ã_{j,t,L_i}       … (4)  절편 제외(원문 명시)
    E_t[β_{i,t+1}] = (1/12) Σ_{m=1..12} β_{i,t+1-m}       … (5)  과거 12개월 평균

  · 래그 11종 **3·5·10·20·50·100·200·400·600·800·1000일** (원문 §2.2 그대로).
  · 예측 기대수익으로 **오분위**(원문은 five portfolios). Q5 = 최고 · Q1 = 최저.
  · **Q5 − Q1 zero-investment**, **동일가중**, 1개월 보유, 매월 재정렬.
  · 레짐 필터 없음(원문에 없음).
  · 번인: **첫 1,000일 + 이후 12개월 폐기**(원문 §2.4 그대로).

  ⚠️ **오분위다 — 십분위가 아니다.** registry 등재 시 '십분위'로 잘못 적혔다.
     원문 §2.2: "We sort all stocks into five portfolios by their expected returns".
     각주 5는 십분위/시총가중이면 결과가 "similar (stronger/weaker)"라고만 한다.

원문 Table 1 (1930-06~2014-12, 1,015개월):
     평균 **월 +1.63%p** · 표준편차 3.45% · Sharpe 0.47 · 왜도 1.47 · 첨도 11.3 · **t (15.0)**
     **1.63 은 평균이고 3.45 는 t 가 아니라 표준편차다** (t 는 원문이 따로 인쇄한다).

**검정력·Type M 의 입력은 1.63 이 아니라 1.78** — 이 설계는 가격 하한만 쓰고
사이즈 필터를 못 쓰므로(과거 시총 부재), 대응하는 원문 값은 Table 10 의
`price filter only` = **월 1.78%p** 다. 헤드라인 1.63 은 두 필터를 다 쓴 값이다.

회전율 척도 주의 — 원문 본문의 `65.6%` 는 **다리당 Σ|Δw|** 이고 Table 13 의
`131.2%` 는 **양 다리 총 거래액**이다. 저장소 `turn` 과 같은 척도(다리당 편도)로는
**32.8%/월**이다. 상세는 `LIT_TURNOVER` 주석.

판정은 PROTOCOL §3 개정판(2026-08-12) — **6종 추정량이 갈리면 측정 불가**.

  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/trend_factor.py --selftest | --power | --run

**--power 는 SE·회전율·표본크기만 출력한다. 평균·t·부호는 출력 경로가 없다**
(PROTOCOL §3.1-2 — 점추정치를 보면 사전등록이 무효가 된다).
`--run` 은 **게이트 통과 후** 작성됐고, 여기서 처음으로 평균·t·부호가 나온다.
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading

sys.path.insert(0, "docs/analysis/backtests/scripts")

_N = NormalDist()

# ── 원문이 정한 것 (바꾸지 않는다) ─────────────────────────────────────────
LAGS = (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000)   # 원문 §2.2
N_QUINT = 5                                                 # 원문 §2.2 "five portfolios"
BETA_WIN = 12                                               # 원문 식 (5)
# 번인은 **별도 상수로 자르지 않는다** — `eligible()` 이 래그 11종 전부 유효를
# 요구하므로 최장 래그가 곧 번인이다. 원문(§2.4 "첫 1,000일 폐기")보다 엄격하다:
# 표본 앞 1,000일이 아니라 **종목별로 직전 1,000봉이 연속 유효**해야 한다.
BURN_DAYS = max(LAGS)                                       # 원문 §2.4 = 1,000일

# 원문 Table 1 · 8 · 10 · 13 — **사전 고정**
LIT_HEADLINE = 1.63        # %p/월, Q5−Q1, 헤드라인 (가격 하한 ∧ 사이즈 필터 **둘 다**)
LIT_SD = 3.45              # %/월 — **t 값이 아니다**. 원문 Table 1 은 t 를 (15.0) 로 인쇄
LIT_T = 15.0               # 원문 Table 1 인쇄값
LIT_SHARPE = 0.47

# **이 설계에 대응하는 원문 값** — 가격 하한 O · 사이즈 필터 X (§2 이탈 #1).
#   원문 Table 10 / 본문 784행: `if we impose only the price filter, the average
#   return is 1.78%, the volatility is 3.37%, and the Sharpe ratio is 0.53`
LIT_SPREAD = 1.78          # ← 검정력·Type M 의 입력. 헤드라인 1.63 이 아니다.

# 롱온리 부수 판정용 — 원문 Table 8 (오분위 평균수익)에서 유도.
#   Q5=1.93 · Q1=0.31 (본문 690·692행) · Q2=Q1+0.56=0.87 · Q4=Q5−0.53=1.40 (696-697행)
#   Q3 는 단조성으로 (0.87, 1.40) 안 → 유니버스 평균 1.08~1.19 → Q5−평균 0.74~0.85
LIT_LONGONLY = 0.80        # 중앙값. **원문 미보고 값에서 유도했음을 문서에 명기**

# ── 원문 회전율 척도 (게이트 2차가 잡은 결함 — 초안이 2배 잘못 읽었다) ──────
#
# 원문 본문 950행 `The turnover rate of the trend factor is 65.6%` 와
# Table 13 `Turnover(%) Mean = 131.2` 는 **정확히 2배 관계**이며, 그 2배는 **양 다리**다.
# Table 13 의 Turnover 열은 BETC 를 곱하면 수익이 되는 **거래액 베이스**다:
#     Panel A  1.24 × 131.2% = 1.627 ≈ 1.63  (트렌드)
#     Panel B  0.68 ×  75.1% = 0.511          (모멘텀)
#     Panel C  1.99 ×  56.1% = 1.116 = 1.627 − 0.511   ← 56.1 = 131.2 − 75.1
# 팩터가 $1 롱 + $1 숏이므로 131.2% = 양 다리 총 거래액 = 4 × (다리당 편도 회전율).
#   ⇒ 원문 "회전율 65.6%" = **다리당 Σ|Δw|** = 2 × 편도
#   ⇒ 원문 **다리당 편도 회전율 = 32.8%/월**
# 교차검증: 모멘텀 37.6% → 편도 18.8%/월. J-T(1993) 6개월 보유의 이론값 1/6 = 16.7%와
# 정합한다. "37.6%가 편도"라면 이론값의 2.25배가 되어 맞지 않는다.
LIT_TURNOVER_SIGMA = 0.656    # 원문 표기 — **다리당 Σ|Δw|**
LIT_TURNOVER = 0.328          # ← 저장소 `turn` 과 같은 척도(**다리당 편도**)
LIT_T13_BASE = 1.312          # Table 13 Turnover 열 = 양 다리 총 거래액
LIT_BETC = 1.24               # %/월 — 수익을 0으로 만드는 **달러 거래액당** 요율
LIT_MOM_SIGMA, LIT_MOM_T13, LIT_MOM_BETC, LIT_MOM_RET = 0.376, 0.751, 0.68, 0.511
LIT_DIR = +1               # **문헌이 예측하는 부호** — 양수 유의면 채택

# ── 저장소 규약 (PROTOCOL §2 · 실행 위생이라 §0.3에서 유지) ────────────────
PRICE_FLOOR = {"kr": 1000.0, "us": 5.0}    # 원문 $5 그대로 · KR 은 저장소 대응물
DV_WIN = 20
DV_MIN = {"kr": None, "us": 2e6}           # PROTOCOL §2 (PIT) · **KR 은 하한 없음**
# **편도 요율.** PROTOCOL §2 의 왕복(KR 0.28% · US 0.10%)의 절반이다 —
# `_turn_cost` 가 곱하는 `Σ|Δw|` 는 전량 교체 시 2.0(= 편도 회전율의 2배)이라
# 편도 요율을 곱해야 왕복 1회에 왕복 요율이 물린다.
# 선례 3건(H-025·H-026·H-028)이 모두 같은 규약이다.
# 원문도 **구조는 같다**(요율 × 거래액). 단 원문의 거래액 베이스는 양 다리 합계이고
# 저장소 `turn` 은 다리당 편도라 **척도가 다르다** — 비교할 때 반드시 환산할 것.
COST = {"kr": 0.0014, "us": 0.0005}        # 편도
RET_CAP = 1.0                              # 종목-월 수익 상한 (§3.4 견고성 축)
SMA_BENCH = 200                            # 레짐은 원문에 없다 — 보고용으로만 계산

# 다중비교 가족 — **결과를 보기 전에 고정**.
#   ① 롱숏 Q5−Q1 (주 판정)  ② 롱온리 Q5−유니버스 (부수 판정, KR 공매도 금지 대응)
FAMILY = 2
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414

SEED = 20260818                            # 플라시보 P3 난수 시드

# §3.4 **근사 사이즈 필터** 견고성 축의 N — 사전등록은 "PIT 거래대금 상위 N"까지만
# 정하고 N 을 고정하지 않았다. **결과를 보기 전에** 저장소 기존 규약값을 그대로 쓴다:
#   CAP_N  = 베이스라인 C 유니버스 크기(H-020~H-024) · `max_lottery_deciles.CAP_N` 와 동일
#   CAP_N2 = 그 절반 — H-028 이 문턱 배수 ×1.0/×1.5 로 **단조성**을 본 것과 같은 목적
# **판정 대체가 아니라 병기**이므로 둘 다 출력한다(고르지 않는다).
CAP_N = {"kr": 400, "us": 500}
CAP_N2 = {"kr": 200, "us": 250}

# **리밸일 우연의 CAGR 표준편차** — H-016 실측(`2026-08-11-rebalance-timing-luck.md`
# 79·88행). PROTOCOL §3.1-6: "4.02%p(KR) 미만의 CAGR 차이는 우위의 근거로 쓰지 않는다."
# **차분(Q5−Q1)에는 해당 없다**(양 팔이 같은 리밸일이라 상쇄) — 절대 성과 보고용이다.
# ⚠️ 변형이 4개뿐이라 자유도 3의 추정치다(H-016 한계 2) — "대략 4%p 수준"으로만 읽는다.
TIMING_LUCK_SD = {"kr": 4.02, "us": 1.43}


# ────────────────────────────────────────────────────────────── 패널

def _rolling_mean_at(rows: np.ndarray, X: np.ndarray, V: np.ndarray, L: int) -> np.ndarray:
    """`rows` 각 행 i 에서 **직전 L 봉(i 포함)** 종가의 단순평균 — 원문 식 (1).

    누적합으로 O(일수×종목) 1회에 끝낸다. `V` 는 유효 마스크이고
    **L 봉이 전부 유효할 때만** 값을 낸다(부분 창을 평균하지 않는다).
    """
    S = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    K = np.concatenate([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)], axis=0)
    hi = rows + 1
    lo = hi - L
    out = np.full((len(rows), X.shape[1]), np.nan)
    ok = lo >= 0
    if not ok.any():
        return out
    h, l = hi[ok], lo[ok]
    tot = S[h] - S[l]
    cnt = K[h] - K[l]
    full = cnt == L
    seg = np.where(full, tot / L, np.nan)
    out[ok] = seg
    return out


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    cl_raw = piv["close"].to_numpy(float)
    op_raw = piv["open"].to_numpy(float)

    # **신호용 종가** — `close > 0` 만 본다.
    #   KR 거래정지 봉은 open=high=low=0 이고 **close 만 직전가로 이월**돼 있다
    #   (PROTOCOL §1). CRSP 도 무거래일에 직전가/호가중간값을 남기므로,
    #   이월 종가를 이동평균에 쓰는 것이 원문에 가장 가깝다.
    sig_ok = np.isfinite(cl_raw) & (cl_raw > 0)
    Xs = np.where(sig_ok, cl_raw, 0.0)

    # **체결·수익률용 종가** — 거래 가능한 봉만 (PROTOCOL §1: open<=0 도 마스킹)
    trade_ok = sig_ok & np.isfinite(op_raw) & (op_raw > 0)
    cl_tr = np.where(trade_ok, cl_raw, np.nan)

    mk = pd.PeriodIndex(pd.to_datetime(idx), freq="M")
    me = pd.Series(np.arange(len(idx)), index=mk).groupby(level=0).last().to_numpy()
    me = me[:-1]                                  # 마지막 역월은 부분 월 → 제외
    months = pd.PeriodIndex(mk[me])

    # 정규화 분모 P_{j,d} = 월말 종가 (거래 가능한 봉만)
    pm = cl_tr[me]
    mret = np.full_like(pm, np.nan)
    mret[1:] = pm[1:] / pm[:-1] - 1.0             # 월수익 (월말→월말)

    # Ã_{j,t,L} = A/P  — 원문 식 (2). 분모는 **신호용 종가**(원문은 같은 종가)
    pm_sig = np.where(sig_ok[me], cl_raw[me], np.nan)
    At = np.stack([_rolling_mean_at(me, Xs, sig_ok, L) for L in LAGS], axis=-1)
    with np.errstate(all="ignore"):
        A = At / pm_sig[:, :, None]               # (월, 종목, 래그)
    A[~np.isfinite(A)] = np.nan

    dv_d = piv["close"] * piv["volume"]
    pit_dv = dv_d.rolling(DV_WIN, min_periods=DV_WIN).mean().to_numpy(float)[me]

    # 변동성 — 플라시보 P1·P2 전용. 월말 기준 직전 250 거래일 일간수익률 σ
    # **`close > 0` 마스크를 먼저 건다** (PROTOCOL §1 — US 캐시에 종가 0 구간이 있다)
    cl_sig = np.where(sig_ok, cl_raw, np.nan)
    dret = np.full_like(cl_raw, np.nan)
    with np.errstate(all="ignore"):
        dret[1:] = cl_sig[1:] / cl_sig[:-1] - 1.0
    dret[~np.isfinite(dret)] = np.nan
    sig = pd.DataFrame(dret).rolling(250, min_periods=200).std().to_numpy(float)[me]

    b = loading.load_bench(market).set_index("date")["close"].reindex(idx).ffill()
    bv = b.to_numpy(float)
    sma = pd.Series(bv).rolling(SMA_BENCH, min_periods=SMA_BENCH).mean().to_numpy()

    return {"tickers": np.asarray(cols, dtype=str), "months": months, "me": me,
            "A": A, "pm": pm, "mret": mret, "pit_dv": pit_dv, "sigma": sig,
            "close_tr": cl_tr, "n_days": len(idx),
            "bull": (bv[me] > sma[me]), "market": market}


# ──────────────────────────────────────────────────────── 적격 · 회귀

def eligible(P: dict, m: int, *, cap_n: int | None = None) -> np.ndarray:
    """월 m 말 기준 적격 — Ã 11종 전부 유효 + 가격 하한 + PIT 거래대금.

    `cap_n` 은 §3.4 **근사 사이즈 필터** 견고성 축 — PIT 거래대금 상위 N만 남긴다.
    과거 시점 시총이 없어(§2 이탈 #1) 거래대금을 규모 대리변수로 쓴다.
    H-028이 같은 대리변수로 재서 추정치가 −24% 움직인 전례가 있다.
    **주 판정은 `cap_n=None`.**
    """
    mkt = P["market"]
    ok = np.isfinite(P["A"][m]).all(axis=1)
    px = P["pm"][m]
    ok &= np.isfinite(px) & (px >= PRICE_FLOOR[mkt])
    dv = P["pit_dv"][m]
    if DV_MIN[mkt] is not None:
        ok &= np.isfinite(dv) & (dv >= DV_MIN[mkt])
    if cap_n is not None:
        ok &= np.isfinite(dv)
        cand = np.flatnonzero(ok)
        if len(cand) > cap_n:
            thr = np.sort(dv[cand])[::-1][cap_n - 1]
            ok &= dv >= thr
    return ok


def cross_section_beta(P: dict, m: int) -> tuple[np.ndarray, int, float]:
    """원문 식 (3) — 월 m 수익률을 **월 m−1 말** Ã 에 회귀. 반환 (β 11개, n, cond).

    절편을 포함해 추정하고 **β_0 는 버린다**(식 (4)가 절편을 제외한다).
    11개 Ã 는 전부 1 근처의 비율이라 심하게 공선이다 — 원문이 그대로 하므로
    그대로 두되 `lstsq`(SVD)로 풀고 조건수를 진단으로 남긴다.
    """
    nan = (np.full(len(LAGS), np.nan), 0, np.nan)
    if m < 1:
        return nan
    ok = eligible(P, m - 1) & np.isfinite(P["mret"][m])
    n = int(ok.sum())
    if n < len(LAGS) + 2:
        return nan
    X = np.column_stack([np.ones(n), P["A"][m - 1][ok]])
    y = P["mret"][m][ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cond = float(np.linalg.cond(X))
    return beta[1:], n, cond


def forecast(P: dict) -> dict:
    """식 (5)→(4). 월 m 말에 만드는 예측은 월 m+1 수익률용이다.

    `fc[m]` = 월 m 말 기준 각 종목의 E[r_{m+1}]. 번인(1,000일 + 12개월) 전은 NaN.
    """
    n_m = len(P["months"])
    n_j = P["A"].shape[1]
    B = np.full((n_m, len(LAGS)), np.nan)
    ns = np.zeros(n_m, dtype=int)
    conds = np.full(n_m, np.nan)
    for m in range(n_m):
        B[m], ns[m], conds[m] = cross_section_beta(P, m)

    fc = np.full((n_m, n_j), np.nan)
    eb = np.full((n_m, len(LAGS)), np.nan)
    for m in range(n_m):
        # **음수 슬라이스 가드** — m < 11 이면 `B[음수:m+1]` 가 꼬리를 집어
        # 조용히 **미래 β 를 섞는다**. 현재 표본에서는 빈 슬라이스라 무해하나
        # 표본이 짧아지면 look-ahead 가 된다 (게이트 1차 지적).
        if m < BETA_WIN - 1:
            continue
        win = B[m - BETA_WIN + 1: m + 1]                # β_{m-11} … β_m (12개)
        if len(win) < BETA_WIN or not np.isfinite(win).all():
            continue
        e = win.mean(axis=0)
        eb[m] = e
        fc[m] = P["A"][m] @ e                            # 절편 없음 — 원문 식 (4)
    return {"beta": B, "n_reg": ns, "cond": conds, "fc": fc, "ebeta": eb}


# ──────────────────────────────────────────────────────── 오분위 · 실행

def split_quintiles(score: np.ndarray, ok: np.ndarray, tickers: np.ndarray) -> list:
    """예측 **오름차순**(동점은 **티커 오름차순**) 5등분. [0]=Q1(최저) … [4]=Q5(최고).

    적격 수가 5의 배수가 아니면 `np.array_split` 규약대로 **나머지가 앞쪽 그룹
    (=Q1 쪽)에 붙는다** — 즉 Q1 이 Q5 보다 최대 1종목 많을 수 있다.
    """
    cand = np.flatnonzero(ok & np.isfinite(score))
    if len(cand) < N_QUINT * 2:
        return []
    order = np.array(sorted(cand, key=lambda j: (score[j], tickers[j])), dtype=int)
    return [np.asarray(g, dtype=int) for g in np.array_split(order, N_QUINT)]


def _hold_return(P: dict, m: int, mem: np.ndarray, *, exec_lag: int = 0,
                 ret_cap: float | None = None) -> float:
    """월 m 말 편입 → 월 m+1 수익률. **원문은 월말 종가 체결**(`exec_lag=0`).

    `exec_lag=1` 은 §3.4 견고성 축(T 종가 판정 → T+1 종가 체결).
    `ret_cap` 은 종목-월 수익 상한(§3.4). 둘 다 **주 판정에는 쓰지 않는다.**
    보유 종목 중 **다음 달 수익이 NaN 인 종목은 동일가중 평균에서 탈락**시킨다
    (상장폐지·장기정지). 전원 탈락이면 0을 반환한다.
    """
    if len(mem) == 0:
        return 0.0
    if exec_lag == 0:
        r = P["mret"][m + 1][mem]
    else:
        i0, i1 = P["me"][m] + exec_lag, P["me"][m + 1] + exec_lag
        if i1 >= P["n_days"]:
            return np.nan
        c = P["close_tr"]
        with np.errstate(all="ignore"):
            r = c[i1][mem] / c[i0][mem] - 1.0
    r = r[np.isfinite(r)]
    if ret_cap is not None:
        r = np.minimum(r, ret_cap)
    return float(r.mean()) if len(r) else 0.0


def _turn_cost(prev, mem, drift, cost):
    """직전 보유가 수익률로 표류한 뒤의 비중 대비 회전율 × 비용."""
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


def run_arm(P: dict, F: dict, *, score: np.ndarray | None = None,
            exec_lag: int = 0, ret_cap: float | None = None,
            cap_n: int | None = None) -> dict:
    """오분위 팔을 돌린다. `score=None` 이면 본안(트렌드 예측).

    `exec_lag`·`ret_cap`·`cap_n` 은 §3.4 견고성 축이며 **주 판정은 전부 기본값**이다.
    """
    n_m = len(P["months"])
    cost = COST[P["market"]]
    sc_all = F["fc"] if score is None else score
    prev = [None] * N_QUINT
    rows, gross, costs, turns, months, univ, nq, eln = [], [], [], [], [], [], [], []
    hr = lambda mm, mem: _hold_return(P, mm, mem, exec_lag=exec_lag, ret_cap=ret_cap)
    for m in range(n_m - 1):
        sc = sc_all[m]
        if not np.isfinite(sc).any():
            continue
        ok = eligible(P, m, cap_n=cap_n)
        qs = split_quintiles(sc, ok, P["tickers"])
        if not qs:
            continue
        drift = np.nan_to_num(P["mret"][m], nan=0.0)
        g = np.zeros(N_QUINT)
        c = np.zeros(N_QUINT)
        t = 0.0
        for q in range(N_QUINT):
            g[q] = hr(m, qs[q])
            cc, prev[q] = _turn_cost(prev[q], qs[q], drift, cost)
            c[q] = cc
            if q in (0, N_QUINT - 1) and cost > 0:
                t += cc / cost / 2.0 / 2.0        # 두 다리 **평균 편도** 회전율
        u = np.flatnonzero(ok)
        rows.append((g - c) * 100.0)
        gross.append(g * 100.0)
        costs.append(c * 100.0)
        turns.append(t)
        univ.append(hr(m, u) * 100.0)
        nq.append(np.mean([len(x) for x in qs]))
        eln.append(len(u))
        months.append(m)
    if not rows:
        return {}
    R, G, C = np.asarray(rows), np.asarray(gross), np.asarray(costs)
    spread = (G[:, N_QUINT - 1] - G[:, 0]) - (C[:, N_QUINT - 1] + C[:, 0])
    longonly = R[:, N_QUINT - 1] - np.asarray(univ)
    return {"q": R, "gross": G, "cost": C, "spread": spread, "longonly": longonly,
            "uni": np.asarray(univ), "turn": np.asarray(turns),
            "months": np.asarray(months), "n": len(rows),
            "nq": float(np.mean(nq)), "elig": float(np.mean(eln))}


# ────────────────────────────────────────────────────────── 플라시보

def placebo_scores(P: dict, F: dict) -> dict:
    """§3.3 사전 배치 플라시보 — **결과를 보기 전에 고정했다.**

    P1 σ 정렬     : 250일 일간수익률 표준편차로 직접 정렬
    P2 σ 직교화   : E[r] 을 매월 σ 에 횡단면 회귀한 **잔차**로 정렬
    P3 계수 무작위: E[β] 를 **시드 고정 무작위 월**의 β 로 교체(시점 정보만 파괴)
    P4 규모 정렬  : PIT 거래대금으로 정렬 (H-018 교훈 — 신호가 규모 정렬일 수 있다)
    """
    n_m, n_j = F["fc"].shape
    rng = np.random.default_rng(SEED)

    p1 = P["sigma"].copy()

    p2 = np.full_like(F["fc"], np.nan)
    for m in range(n_m):
        f, s = F["fc"][m], P["sigma"][m]
        ok = eligible(P, m) & np.isfinite(f) & np.isfinite(s)
        if ok.sum() < 10:
            continue
        X = np.column_stack([np.ones(int(ok.sum())), s[ok]])
        b, *_ = np.linalg.lstsq(X, f[ok], rcond=None)
        p2[m, ok] = f[ok] - X @ b

    # P3 — β 행이 전부 유효한 월만 후보로 두고, 매월 그중 하나를 무작위로 뽑아 쓴다
    valid = np.flatnonzero(np.isfinite(F["beta"]).all(axis=1))
    p3 = np.full_like(F["fc"], np.nan)
    if len(valid) >= BETA_WIN:
        for m in range(n_m):
            if not np.isfinite(F["fc"][m]).any():
                continue
            pick = rng.choice(valid, size=BETA_WIN, replace=False)
            p3[m] = P["A"][m] @ F["beta"][pick].mean(axis=0)

    p4 = P["pit_dv"].copy()
    return {"P1 σ정렬": p1, "P2 σ직교화": p2, "P3 계수무작위": p3, "P4 규모정렬": p4}


# ─────────────────────────────────────────────────────────────── 명령

def _pc():
    import pooled_clustering as PC
    return PC


_PANELS: dict = {}


def _panels() -> dict:
    if not _PANELS:
        for mk in ("kr", "us"):
            P = build_panel(mk)
            _PANELS[mk] = (P, forecast(P))
    return _PANELS


def cmd_power():
    PC = _pc()
    print("=" * 100)
    print("[H-033] 사전 검출력 — **SE·회전율·표본만**. 평균·t·부호는 출력 경로가 없다.")
    print(f"  가족 {FAMILY}칸 Bonferroni → |t| > {K_CRIT:.4f}"
          f" · 문헌 효과 월 +{LIT_SPREAD}%p · 문헌 회전율 {LIT_TURNOVER*100:.1f}%/월")
    print("  검정력은 PROTOCOL §3 개정판대로 **6종 중 SE 최대(= 최저 검정력)** 기준.")
    print("=" * 100)
    ser, lo_ser = {}, {}
    cost_n, cost_w, lo_cost_n, lo_cost_w = 0.0, 0, 0.0, 0
    for mk, (P, F) in _panels().items():
        a = run_arm(P, F)
        if not a:
            print(f"\n── {mk.upper()} ── 산출 불가 (번인 후 남는 달 없음)")
            continue
        mm = P["months"][a["months"]]
        ser[mk] = pd.Series(a["spread"], index=mm)
        lo_ser[mk] = pd.Series(a["longonly"], index=mm)
        cs = a["cost"]
        cm = float(cs[:, N_QUINT - 1].mean() + cs[:, 0].mean())
        cost_n += cm * a["n"]; cost_w += a["n"]
        lo_cost_n += float(cs[:, N_QUINT - 1].mean()) * a["n"]; lo_cost_w += a["n"]
        ok_reg = np.isfinite(F["cond"])
        print(f"\n── {mk.upper()} ── n={a['n']}개월 ({mm[0]}~{mm[-1]})"
              f" · 적격 평균 {a['elig']:.0f}종목 · 오분위 {a['nq']:.1f}종목")
        tn = a["turn"].mean()
        print(f"   회전율(다리당 편도) {tn*100:.1f}%/월"
              f" — 원문 {LIT_TURNOVER*100:.1f}% 의 **{tn/LIT_TURNOVER:.2f}배**"
              f" (원문 표기 {LIT_TURNOVER_SIGMA*100:.1f}% 는 다리당 Σ|Δw|)")
        print(f"   스프레드 월 비용 {cm:.4f}%p → 문헌 효과의 {cm/LIT_SPREAD*100:.1f}% 잠식")
        print(f"   횡단면 회귀 {int(ok_reg.sum())}개월 · 회귀 표본 중위"
              f" {int(np.median(F['n_reg'][ok_reg])) if ok_reg.any() else 0}종목"
              f" · 조건수 중위 {np.nanmedian(F['cond']):.3g}")
        print(f"   naive SE {PC.naive_se({'x': ser[mk]})[1]:.4f}%p (참고)")
    if len(ser) < 2:
        print("\n두 시장 모두 산출되지 않아 통합 검정력을 낼 수 없다.")
        return
    cells = (("① 롱숏 Q5−Q1", ser, cost_n / max(cost_w, 1), LIT_SPREAD,
              "원문 Table 10 `price filter only` = 이 설계의 필터 구성"),
             ("② 롱온리 Q5−유니버스", lo_ser, lo_cost_n / max(lo_cost_w, 1),
              LIT_LONGONLY, "원문 Table 8 오분위에서 **유도** — 원문 미보고"))
    for tag, s, cst, lit, src in cells:
        est = PC.all_estimates(s)
        ses = {k: est[k] for k in PC.VOTERS}
        worst = max(ses, key=lambda k: ses[k])
        net = lit - cst                              # **비용 반영 후** (PROTOCOL §3.1-2)
        d = net / est[worst]
        pw = _N.cdf(d - K_CRIT) + _N.cdf(-d - K_CRIT)
        dg = lit / est[worst]
        pwg = _N.cdf(dg - K_CRIT) + _N.cdf(-dg - K_CRIT)
        print(f"\n[통합 {tag}] n={est['n']}개월 · 겹치는 달 {est['overlap']}"
              f" · ρ={est['rho']:+.4f}")
        print(f"   G(달)={est['G_달']} G(분기)={est['G_분기']} G(연)={est['G_연']}")
        for k in PC.VOTERS:
            print(f"   {k:<10} SE {est[k]:.4f}%p")
        print(f"   naive      SE {est['naive']:.4f}%p (판정에 투표하지 않음)")
        print(f"   문헌 효과 {lit}%p/월 — {src}")
        print(f"   표본가중 비용 {cst:.4f}%p/월 → {lit} − {cst:.4f}"
              f" = **net {net:.4f}%p**")
        print(f"   **SE 최대 = {worst} ({est[worst]:.4f}) → 검정력(비용 후) = {pw*100:.1f}%**"
              f"  (gross 였다면 {pwg*100:.1f}%)")


# ──────────────────────────────────────────────── 판정 보조 (--run 전용)

def tstat(x) -> float:
    """naive t — **보고용**. 판정은 6종 클러스터/DK 추정량이 한다."""
    a = np.asarray(x, dtype=float)
    s = a.std(ddof=1)
    return float(a.mean() / (s / np.sqrt(len(a)))) if s > 0 else float("nan")


def spearman(a, b) -> float:
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def cagr(monthly_pct: np.ndarray) -> float:
    """월 %수익 계열 → 연 %CAGR."""
    g = np.prod(1.0 + np.asarray(monthly_pct, dtype=float) / 100.0)
    return float((g ** (12.0 / len(monthly_pct)) - 1.0) * 100.0)


def type_m(se: float, effect: float, k: float = K_CRIT) -> float:
    """`E[|d̂| | 유의] / effect` — 유의 시 과대추정 배율(§3.5). **닫힌 형태**.

    `z ~ N(d, 1)` (d = effect/se) 에서 유의 영역은 `z < −k` 와 `z > +k` 의
    **서로 떨어진 두 구간**이다. 하나로 이어 적분하면 `(−k, +k)` 를 가로지르는
    가짜 구간이 분자·분모에 섞인다 (H-026 게이트 3회차가 잡은 오류).

        P(|z| > k)        = Φ(d − k) + Φ(−d − k)
        E[ z·1{z > k}]    =  d·Φ(d − k) + φ(k − d)
        E[−z·1{z < −k}]   = −d·Φ(−d − k) + φ(k + d)
    """
    d = effect / se
    if d == 0:
        return float("nan")
    phi = lambda t: float(np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi))
    p = _N.cdf(d - k) + _N.cdf(-d - k)
    if p <= 0:
        return float("nan")
    num = (d * _N.cdf(d - k) + phi(k - d)) + (-d * _N.cdf(-d - k) + phi(k + d))
    return float((num / p) / d)


def arm_series(P: dict, F: dict, key: str = "spread", **kw) -> tuple:
    """팔을 돌려 (월별 계열, 원 결과) 를 돌려준다. `key` 는 `spread` 또는 `longonly`."""
    a = run_arm(P, F, **kw)
    if not a:
        return None, None
    return pd.Series(a[key], index=P["months"][a["months"]]), a


def _pooled_cost(arms: dict, longonly: bool = False) -> float:
    """표본가중 월 비용 — `--power` §4.3 과 같은 방식(Type M 입력의 net 효과용)."""
    num = den = 0.0
    for a in arms.values():
        cs = a["cost"]
        cm = float(cs[:, N_QUINT - 1].mean())
        if not longonly:
            cm += float(cs[:, 0].mean())
        num += cm * a["n"]
        den += a["n"]
    return num / den if den else float("nan")


def _report_cell(PC, ser: dict, title: str, lit: float, note: str,
                 cost: float | None = None) -> dict:
    """판정 칸 하나를 6종 추정량으로 판정하고 출력한다. 반환 = `all_estimates`."""
    est = PC.all_estimates(ser)
    rv, vs, agree = PC.robust_verdict(est, K_CRIT, family=FAMILY, direction=LIT_DIR)
    rvt, _, agt = PC.robust_verdict(est, K_CRIT, use_t=True, family=FAMILY,
                                    direction=LIT_DIR)
    worst = max(PC.VOTERS, key=lambda k: est[k])
    print(f"\n{'=' * 104}\n[{title}]  문헌 {lit:+.2f}%p/월 — {note}")
    print(f"  n={est['n']}개월 · 겹치는 달 {est['overlap']} · ρ={est['rho']:+.4f}"
          f" · **월 {est['mu']:+.4f}%p · 연 {est['mu'] * 12:+.2f}%p**")
    print(f"  {'추정량':<10}{'SE':>9}{'t':>9}   판정")
    print(f"  {'naive(참고)':<10}{est['naive']:>9.4f}{est['mu'] / est['naive']:>9.3f}"
          f"   {PC.verdict(est['mu'] / est['naive'], K_CRIT, LIT_DIR)}  ← 투표하지 않음")
    for k in PC.VOTERS:
        print(f"  {k:<10}{est[k]:>9.4f}{est['mu'] / est[k]:>9.3f}   {vs[k]}")
    print(f"  → 6종 {'일치' if agree else '**갈림**'} · **판정 = {rv}**"
          f"   (direction={LIT_DIR:+d} · 임계 {K_CRIT:.4f})")
    print(f"  → t(df) 참조분포로 바꾸면: {rvt} ({'일치' if agt else '갈림'})"
          f" — 판정 {'불변' if rvt == rv else '**변동**'}  [견고성 §3.4]")
    lo = est["mu"] - 1.96 * est[worst]
    hi = est["mu"] + 1.96 * est[worst]
    print(f"  95% CI(SE 최대={worst} {est[worst]:.4f}) = [{lo:+.4f}, {hi:+.4f}] %p/월"
          f" · 0 포함? {'예' if lo <= 0 <= hi else '아니오'}"
          f" · 문헌값({lit}) 포함? {'예' if lo <= lit <= hi else '아니오'}")
    if cost is not None:
        net = lit - cost
        pw = _N.cdf(net / est[worst] - K_CRIT) + _N.cdf(-net / est[worst] - K_CRIT)
        print(f"  Type M(설계 단계 · **가정 효과** 기준) = {type_m(est[worst], net):.3f}배 "
              f"(net e* = {lit} − 비용 {cost:.4f} = {net:.4f} · SE 최대"
              f" · 사전 검정력 {pw * 100:.1f}%)")
        # **오독 방지 (§4-(b) 경미 4)** — 위 값은 "문헌 효과가 참일 때" 의 배율이라
        # 1.0 에 가깝다. 결과가 비유의인 지금 그것을 "우리 추정치는 안 부풀었다"로
        # 읽으면 정반대다. **관측 효과 기준 배율을 반드시 병기**한다.
        print(f"  Type M(**관측 효과 {est['mu']:+.4f} 기준**) ="
              f" **{type_m(est[worst], est['mu']):.3f}배**"
              f" — 이 크기의 효과가 참이었다면 유의했을 때 이만큼 부풀어 보인다")
    print(f"  [시장별 — **임계 없이 병기**]")
    signs = []
    for mk, s in ser.items():
        v = s.to_numpy()
        signs.append(np.sign(v.mean()))
        print(f"    {mk.upper()} n={len(v)} · 월 {v.mean():+.4f}%p"
              f" · 연 {v.mean() * 12:+.2f}%p · naive t={tstat(v):+.3f}")
    print(f"    부호 {'**일치**' if len(set(signs)) == 1 else '**불일치 — 판정문에 명기**'}")
    return est


def cmd_run():
    PC = _pc()
    print("=" * 104)
    print("[H-033] 트렌드 팩터 — Han·Zhou·Zhu (2016, JFE) 원문 형태 · H-029 재시도")
    print("  래그 11종 Ã → 월별 횡단면 회귀 → E[β] 12개월 평균 → 오분위")
    print("  Q5 − Q1 zero-investment · 동일가중 · 1개월 보유 · 레짐 없음(원문에 없음)")
    print(f"  판정: 통합 표본 · 가족 {FAMILY}칸 → |t| > {K_CRIT:.4f}"
          f" · **6종 갈리면 측정 불가**(PROTOCOL §3) · direction={LIT_DIR:+d}")
    print(f"  문헌 ① Q5−Q1 = +{LIT_SPREAD}%p/월 (Table 10 `price filter only`)"
          f" · ② 롱온리 = +{LIT_LONGONLY}%p/월 (Table 8 유도 · 원문 미보고)")
    print("=" * 104)

    ser, lo_ser, gr_ser, arms = {}, {}, {}, {}
    for mk, (P, F) in _panels().items():
        s, a = arm_series(P, F)
        if a is None:
            print(f"\n── {mk.upper()} ── 산출 불가")
            continue
        arms[mk] = a
        ser[mk] = s
        mm = P["months"][a["months"]]
        lo_ser[mk] = pd.Series(a["longonly"], index=mm)
        gr_ser[mk] = pd.Series(a["gross"][:, N_QUINT - 1] - a["gross"][:, 0], index=mm)
        print(f"\n{'─' * 104}\n── {mk.upper()} ── n={a['n']}개월 ({mm[0]}~{mm[-1]})"
              f" · 적격 평균 {a['elig']:.0f}종목 · 오분위 {a['nq']:.1f}종목")
        print(f"{'오분위':>7}{'월수익(net)':>13}{'연환산':>10}{'gross':>10}{'비용':>9}")
        for q in range(N_QUINT):
            print(f"{'Q' + str(q + 1):>7}{a['q'][:, q].mean():>+12.4f}%"
                  f"{a['q'][:, q].mean() * 12:>+9.2f}%"
                  f"{a['gross'][:, q].mean():>+9.4f}%{a['cost'][:, q].mean():>8.4f}%")
        sp = a["spread"]
        ann, vol = sp.mean() * 12, sp.std(ddof=1) * np.sqrt(12)
        print(f"  Q5−Q1  월 {sp.mean():+.4f}%p · 연 {ann:+.2f}%p"
              f" · naive t={tstat(sp):+.3f} · 변동성 {vol:.2f}%"
              f" · Sharpe {ann / vol:+.2f} · P(>0) {(sp > 0).mean() * 100:.0f}%")
        print(f"  [임계 없이 보고] 오분위 순위상관 ρ ="
              f" {spearman(np.arange(N_QUINT), a['gross'].mean(axis=0)):+.3f}"
              f"  (원문 Table 8 오분위는 단조 증가 → ρ=+1.000)")
        print(f"  롱온리 Q5−유니버스 월 {a['longonly'].mean():+.4f}%p"
              f" · naive t={tstat(a['longonly']):+.3f}"
              f"   · 유니버스 EW 월 {a['uni'].mean():+.4f}%")
        print(f"  회전율(다리당 편도) {a['turn'].mean() * 100:.1f}%/월"
              f" — 원문 {LIT_TURNOVER * 100:.1f}% 의 {a['turn'].mean() / LIT_TURNOVER:.2f}배")
        # **절대 성과는 리밸일 산포와 함께만 읽는다** (PROTOCOL §3.1-6 · §4-(b) 중대 2).
        # 차분(Q5−Q1)은 양 팔이 같은 리밸일이라 상쇄되지만, CAGR 은 절대 성과다.
        q5c, unic = cagr(a["q"][:, N_QUINT - 1]), cagr(a["uni"])
        noise = TIMING_LUCK_SD[mk]
        gap = q5c - unic
        print(f"  [절대 성과 — **리밸일 산포와 함께만 읽는다**]"
              f" Q1 CAGR {cagr(a['q'][:, 0]):+.2f}% · Q5 CAGR {q5c:+.2f}%"
              f" · 유니버스 CAGR {unic:+.2f}%")
        print(f"     Q5 − 유니버스 = {gap:+.2f}%p vs {mk.upper()} 리밸일 잡음"
              f" {noise:.2f}%p (H-016) →"
              f" {'**잡음 미만 — 우위의 근거로 쓸 수 없다**(PROTOCOL §3.1-6)' if abs(gap) < noise else '잡음 초과'}")

    if len(ser) < 2:
        print("\n두 시장이 모두 산출되지 않아 통합 판정을 낼 수 없다.")
        return

    est1 = _report_cell(PC, ser, "① 주 판정 — 통합 Q5 − Q1 (양 다리 비용 차감)",
                        LIT_SPREAD, "원문 Table 10 `price filter only`",
                        cost=_pooled_cost(arms))
    _report_cell(PC, gr_ser, "[병기] 비용 전(gross) Q5 − Q1 — 원문과 직접 비교 가능",
                 LIT_SPREAD, "판정에 쓰지 않는다")
    _report_cell(PC, lo_ser, "② 부수 판정 — 통합 롱온리 Q5 − 유니버스EW",
                 LIT_LONGONLY, "원문 Table 8 오분위에서 **유도** — 원문 미보고",
                 cost=_pooled_cost(arms, longonly=True))

    # ── §3.3 플라시보 — **귀속 조건은 결과를 보기 전에 고정됐다** ──────────────
    print(f"\n{'=' * 104}\n[③ 플라시보 — §3.3 신호 귀속 조건 (사전 고정)]")
    print("  H-028 이 정확히 여기서 죽었다 — 통합 t 가 6/6 유의했는데 신호를 귀속하지 못했다.")
    # **정확한 사유** (§4-(b) 참고 지적): 1,000봉 번인은 `eligible()` 을 통해 **모든 팔에
    # 공통**으로 걸린다. P1·P4 가 12개월 긴 것은 **E[β] 12개월 창이 없기 때문**이다.
    print("  ⚠️ P1·P4 는 E[β] 12개월 창이 필요 없어 **본안보다 정확히 12개월 길다**"
          " (1,000봉 번인은 `eligible()` 로 전 팔 공통).")
    print("  §3.3 조건 3 은 '**본안 스프레드**의 절반'과 비교하므로 **본안 표본으로 맞춘 값**"
          "으로 판정하고, 전체 표본값을 함께 적는다.")
    pl_est = {}
    for name in ("P1 σ정렬", "P2 σ직교화", "P3 계수무작위", "P4 규모정렬"):
        pser, mser = {}, {}
        for mk, (P, F) in _panels().items():
            sc = placebo_scores(P, F)[name]
            s, a = arm_series(P, F, score=sc)
            if a is None:
                continue
            pser[mk] = s
            mser[mk] = s[s.index.isin(ser[mk].index)]     # **본안 표본으로 정렬**
        if len(pser) < 2 or any(len(v) < 12 for v in mser.values()):
            print(f"\n  {name:<12} 산출 불가 (표본 부족)")
            pl_est[name] = None
            continue
        e = PC.all_estimates(pser)
        em = PC.all_estimates(mser)
        rv, _, ag = PC.robust_verdict(e, K_CRIT, family=FAMILY, direction=LIT_DIR)
        rvm, _, agm = PC.robust_verdict(em, K_CRIT, family=FAMILY, direction=LIT_DIR)
        ts = [e["mu"] / e[k] for k in PC.VOTERS]
        tsm = [em["mu"] / em[k] for k in PC.VOTERS]
        pl_est[name] = (e, rv, ag, em, rvm, agm)
        # **라벨 주의** — 아래 `채택`/`기각` 은 `robust_verdict` 의 3분류 문자열일 뿐
        # **플라시보에 대한 것**이다. 가설 판정이 아니다 (§4-(b) 참고 지적).
        print(f"\n  {name:<12} [전체 표본] 월 {e['mu']:+.4f}%p (연 {e['mu'] * 12:+.2f}%p)"
              f" · n={e['n']} · 6종 t {min(ts):+.3f}~{max(ts):+.3f}"
              f" · 유의성 {rv}({'일치' if ag else '갈림'})")
        print(f"  {'':<12} [**본안 표본**] 월 {em['mu']:+.4f}%p"
              f" (연 {em['mu'] * 12:+.2f}%p) · n={em['n']}"
              f" · 6종 t {min(tsm):+.3f}~{max(tsm):+.3f}"
              f" · 유의성 {rvm}({'일치' if agm else '갈림'})")
        print(f"  {'':<12} 본안(①) 대비 **{abs(em['mu']) / abs(est1['mu']) * 100:.1f}%**"
              f" (전체 표본이면 {abs(e['mu']) / abs(est1['mu']) * 100:.1f}%)"
              f" · 시장별(본안 표본) " + " · ".join(
                  f"{m.upper()} {v.mean():+.4f}" for m, v in mser.items()))

    print(f"\n  {'─' * 100}\n  [귀속 성립 조건 — 셋 다 충족해야 트렌드 신호에 귀속한다]")
    cond = {}
    e2 = pl_est["P2 σ직교화"]
    if e2 is None:
        cond[1] = False
        print("  1. P2 산출 불가 → **불충족**")
    else:
        e = e2[0]
        same = np.sign(e["mu"]) == np.sign(est1["mu"])
        allsig = all(abs(e["mu"] / e[k]) > K_CRIT for k in PC.VOTERS)
        cond[1] = bool(same and allsig)
        print(f"  1. P2(σ직교화) 부호 유지 {same} ∧ 6종 전부 유의 {allsig}"
              f" → {'충족' if cond[1] else '**불충족**'}")
    e3 = pl_est["P3 계수무작위"]
    if e3 is None:
        cond[2] = False
        print("  2. P3 산출 불가 → **불충족**")
    else:
        rv3 = e3[1]
        mx = max(abs(e3[0]["mu"] / e3[0][k]) for k in PC.VOTERS)
        r3 = abs(e3[0]["mu"]) / abs(est1["mu"])
        cond[2] = rv3 == "측정 불가"
        print(f"  2. P3(계수 무작위)이 비유의 — 유의성 {rv3} · 6종 |t| 최대 {mx:.3f}"
              f" → {'충족' if cond[2] else '**불충족**'}")
        # **§4-(b) 중대 1** — 조건 2 도 조건 1 과 같은 병을 앓는다. 충족 사유가
        # "효과가 작아서"가 아니라 "검정이 이 크기를 못 잡아서"일 수 있다.
        if cond[2] and r3 >= 0.5:
            print(f"     ⚠️ **이 '충족'은 귀속 지지 근거가 아니다** — P3 점추정치가"
                  f" 본안의 **{r3 * 100:.1f}%**(부호 동일)로 작지 않다.")
            print(f"     충족된 이유는 오직 |t| 최대 {mx:.3f} < {K_CRIT:.4f} 즉"
                  f" **검정이 이 크기를 잡을 검정력이 없기 때문**이다.")
            print(f"     P3 는 β 를 전 표본에서 뽑아 **β 의 무조건 평균을 보존**하므로"
                  f" Ã 횡단면 구조가 그대로 살아남는다 —")
            print(f"     점추정치는 오히려 '시점 정보가 신호가 아니다' 쪽을 가리킨다."
                  f" **결과 문서에 이대로 적는다.**")
    cond[3] = True
    for nm in ("P1 σ정렬", "P4 규모정렬"):
        if pl_est[nm] is None:
            cond[3] = False
            print(f"  3. {nm} 산출 불가 → **불충족**")
            continue
        em = pl_est[nm][3]                      # **본안 표본으로 맞춘 값으로 판정**
        ratio = abs(em["mu"]) / abs(est1["mu"])
        ratio_full = abs(pl_est[nm][0]["mu"]) / abs(est1["mu"])
        signed = em["mu"] / est1["mu"]          # 부호까지 본 비 (참고)
        bad = ratio >= 0.5
        cond[3] = cond[3] and not bad
        print(f"  3. {nm} 크기 {ratio * 100:.1f}% (< 50% 여야 함"
              f" · 전체 표본이면 {ratio_full * 100:.1f}%)"
              f" → {'충족' if not bad else '**불충족**'}")
        if signed < 0:
            # **§4-(b) 경미 1** — 사전등록 문구는 "절반 이상 크기를 **재현**"이다.
            # 반대 부호는 아무것도 재현하지 않는다. **절댓값 비교는 사전등록보다
            # 엄격한 읽기**이며 귀속을 어렵게 하는 방향이다. 그대로 기록한다.
            print(f"     ※ 부호 포함 비는 {signed * 100:+.1f}% — **본안과 반대 부호**다."
                  f" 절댓값으로 비교한 것은 사전등록보다 **엄격한** 읽기이며")
            print(f"       (반대 부호는 '재현'이 아니다) 귀속을 어렵게 하는 방향이다."
                  f" 결과 문서에 명기한다.")
        if 0.4 <= ratio < 0.6:
            print(f"     ※ **임계 50%에서 {abs(ratio - 0.5) * 100:.1f}%p 거리**다 —"
                  f" 표본 정렬 선택(사전등록에 없음)이 이 값을 움직인다. 여유폭을 그대로 적는다.")
    okall = all(cond.values())
    print(f"\n  ⇒ **귀속 {'성립' if okall else '불성립'}**")
    if not okall:
        print("  ⇒ registry·결과 요약에 반드시 명기: **신호를 트렌드에 귀속할 수 없음 —"
              " 이 결과를 Han·Zhou·Zhu 의 지지/반증 증거로 인용 금지**")
    print("  ※ P2 생존은 지지 증거가 아니다(§3.3) — 플라시보는 귀속을 반증하는 장치다.")
    print("  ※ **조건 1 은 주 판정이 유의하지 않으면 구조적으로 충족될 수 없다** —"
          " P2 는 본안에서 σ 성분을 뺀 것이라")
    print("     본안이 6/6 비유의면 P2 도 6/6 유의가 될 수 없다. 조건 1 의 불충족은"
          " **독립적인 정보가 아니다**.")
    print("     귀속 판단에서 실제로 정보를 갖는 것은 **조건 2·3**이다 — 결과 문서에 이대로 적는다.")

    # ── §3.4 견고성 4축 — 판정을 대체하지 않는다 ──────────────────────────────
    print(f"\n{'=' * 104}\n[④ 견고성 — §3.4 사전 지정. **판정을 대체하지 않는다**]")
    print(f"  ⚠️ 사전등록은 근사 사이즈 필터의 **N 을 고정하지 않았다** — CAP_N"
          f" {CAP_N['kr']}/{CAP_N['us']} 은 게이트 후 고정했고")
    print(f"     저장소 기존 상수(`max_lottery_deciles.CAP_N` = 베이스라인 C 유니버스)에"
          f" 앵커했다. CAP_N2 는 그 절반(단조성 확인).")
    print(f"{'축':<34}{'통합 월':>11}{'연':>9}{'6종 t 최소':>11}{'최대':>9}"
          f"{'n':>6}  판정")
    base_t = [est1["mu"] / est1[k] for k in PC.VOTERS]
    print(f"{'(본안) 주 판정':<34}{est1['mu']:>+10.4f}%{est1['mu'] * 12:>+8.2f}%"
          f"{min(base_t):>11.3f}{max(base_t):>9.3f}{est1['n']:>6}"
          f"  {PC.robust_verdict(est1, K_CRIT, family=FAMILY, direction=LIT_DIR)[0]}")
    axes = [("체결 지연 exec_lag=1 (T+1 종가)", lambda mk: dict(exec_lag=1)),
            (f"이상치 상한 ret_cap=+{RET_CAP * 100:.0f}%", lambda mk: dict(ret_cap=RET_CAP)),
            (f"근사 사이즈 필터 상위 {CAP_N['kr']}/{CAP_N['us']}",
             lambda mk: dict(cap_n=CAP_N[mk])),
            (f"근사 사이즈 필터 상위 {CAP_N2['kr']}/{CAP_N2['us']}",
             lambda mk: dict(cap_n=CAP_N2[mk]))]
    flips = []
    for label, kwf in axes:
        rser = {}
        for mk, (P, F) in _panels().items():
            s, a = arm_series(P, F, **kwf(mk))
            if a is not None:
                rser[mk] = s
        if len(rser) < 2:
            print(f"{label:<34}{'산출 불가':>40}")
            flips.append(label)
            continue
        e = PC.all_estimates(rser)
        rv, _, _ = PC.robust_verdict(e, K_CRIT, family=FAMILY, direction=LIT_DIR)
        ts = [e["mu"] / e[k] for k in PC.VOTERS]
        print(f"{label:<34}{e['mu']:>+10.4f}%{e['mu'] * 12:>+8.2f}%"
              f"{min(ts):>11.3f}{max(ts):>9.3f}{e['n']:>6}  {rv}")
        if rv != PC.robust_verdict(est1, K_CRIT, family=FAMILY, direction=LIT_DIR)[0]:
            flips.append(label)
    if flips:
        print(f"\n  ⚠️ **하나만 바꿔도 판정이 뒤집히는 축이 있다 — 판정문에 명기**: "
              + " · ".join(flips))
    else:
        print("\n  → 네 축 전부에서 판정 불변")

    for mk in ("kr", "us"):
        w = loading.staleness_warning(loading.load_prices(mk))
        if w:
            print(f"\n[캐시] {mk.upper()}: {w}")


def cmd_selftest():
    ok = []

    def chk(name, cond):
        # **번호는 자동으로 매긴다** — 손으로 붙인 원문자가 항목 추가 때마다
        # 충돌·중복을 만들었다(게이트 2차에서 실제 발생). 문서는 번호가 아니라
        # 이름으로 인용한다.
        ok.append((f"{len(ok) + 1:02d}", name, bool(cond)))

    # 이동평균 — 손으로 계산 가능한 소형 예제
    X = np.arange(1, 11, dtype=float).reshape(-1, 1)
    V = np.ones_like(X, dtype=bool)
    r = np.array([9])
    chk("MA(L=3) at row9 = mean(8,9,10) = 9",
        np.isclose(_rolling_mean_at(r, X, V, 3)[0, 0], 9.0))
    chk("MA(L=10) at row9 = mean(1..10) = 5.5",
        np.isclose(_rolling_mean_at(r, X, V, 10)[0, 0], 5.5))
    chk("MA(L=11) at row9 = NaN (창 부족)",
        not np.isfinite(_rolling_mean_at(r, X, V, 11)[0, 0]))
    V2 = V.copy(); V2[8] = False
    X2 = X.copy(); X2[8] = 0.0
    chk("창 안에 결측 1개 → NaN (부분 창 평균 금지)",
        not np.isfinite(_rolling_mean_at(r, X2, V2, 3)[0, 0]))

    # Ã 의 차원·극단값 — 가격이 상수면 A/P = 1 (모든 래그)
    Xc = np.full((20, 1), 7.0)
    Vc = np.ones_like(Xc, dtype=bool)
    chk("상수 가격이면 Ã = A/P = 1",
        np.isclose(_rolling_mean_at(np.array([19]), Xc, Vc, 5)[0, 0] / 7.0, 1.0))
    # 단조 상승이면 과거 평균 < 현재가 → Ã < 1
    chk("단조 상승이면 Ã < 1", _rolling_mean_at(r, X, V, 5)[0, 0] / 10.0 < 1.0)
    # 단조 하락이면 Ã > 1
    chk("단조 하락이면 Ã > 1",
        _rolling_mean_at(r, X[::-1].copy(), V, 5)[0, 0] / X[::-1][9, 0] > 1.0)

    # 오분위 분할 — 25개면 5개씩, Q1 이 최저
    sc = np.arange(25, dtype=float)[::-1].copy()          # 큰 값이 앞
    tk = np.array([f"T{i:02d}" for i in range(25)])
    qs = split_quintiles(sc, np.ones(25, dtype=bool), tk)
    chk("오분위 5그룹 × 5종목", len(qs) == 5 and all(len(g) == 5 for g in qs))
    chk("Q1 = 점수 최저 5개", set(qs[0]) == {20, 21, 22, 23, 24})
    chk("Q5 = 점수 최고 5개", set(qs[4]) == {0, 1, 2, 3, 4})
    chk("표본 부족(<10)이면 빈 리스트",
        split_quintiles(sc[:9], np.ones(9, dtype=bool), tk[:9]) == [])

    # 회귀 복원 — 알려진 계수를 정확히 되찾는가
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(200, len(LAGS)))
    btrue = rng.normal(size=len(LAGS))
    yr = 0.5 + Xr @ btrue
    bhat, *_ = np.linalg.lstsq(np.column_stack([np.ones(200), Xr]), yr, rcond=None)
    chk("절편 포함 회귀가 참 계수 복원", np.allclose(bhat[1:], btrue, atol=1e-8))
    chk("절편도 복원(식 (4)에서 버릴 대상)", np.isclose(bhat[0], 0.5, atol=1e-8))

    # 회전율 — 전량 교체면 Σ|Δw| = 2.0 (= 편도 회전율 1.0 의 2배)
    c, _ = _turn_cost({0: 0.5, 1: 0.5}, np.array([2, 3]), np.zeros(4), 1.0)
    chk("전량 교체 시 Σ|Δw| = 2.0 (편도 회전율 1.0)", np.isclose(c, 2.0))
    c2, _ = _turn_cost({0: 0.5, 1: 0.5}, np.array([0, 1]), np.zeros(4), 1.0)
    chk("동일 보유 · 무표류면 회전율 0", np.isclose(c2, 0.0))

    # **비용 요율 규약** — 게이트 1차가 잡은 결함(왕복 요율을 편도 거래에 물림)
    #   편도 회전율 100% = 왕복 1회 → 차감액은 **왕복 요율**과 같아야 한다.
    for mkt, rt in (("kr", 0.0028), ("us", 0.0010)):
        cc, _ = _turn_cost({0: 1.0}, np.array([1]), np.zeros(2), COST[mkt])
        chk(f"{mkt.upper()} 편도 회전율 100% 차감 = 왕복 {rt*100:.2f}%",
            np.isclose(cc, rt))
    chk("COST 는 PROTOCOL §2 왕복의 절반(편도)",
        np.isclose(COST["kr"], 0.0028 / 2) and np.isclose(COST["us"], 0.0010 / 2))
    # **원문 회전율 척도 확정** — Table 13 세 패널의 항등식으로 닫는다.
    #    이것은 저장소 척도를 원문에 맞추는 검증이 아니라, **원문 안에서 131.2%가
    #    무엇의 척도인지**를 확정하는 검증이다 (게이트 2차 지적).
    chk("Panel A: BETC 1.24 × 131.2% = 1.63 (헤드라인)",
        abs(LIT_BETC * LIT_T13_BASE - LIT_HEADLINE) < 0.01)
    chk("Panel B: 0.68 × 75.1% = 0.511 (모멘텀)",
        abs(LIT_MOM_BETC * LIT_MOM_T13 - LIT_MOM_RET) < 0.01)
    chk("Panel C: 56.1 = 131.2 − 75.1 이고 1.99 × 56.1% = (1.24×1.312) − 0.511",
        abs((LIT_T13_BASE - LIT_MOM_T13) - 0.561) < 0.002
        and abs(1.99 * 0.561 - (LIT_BETC * LIT_T13_BASE - LIT_MOM_RET)) < 0.01)
    chk("본문 65.6%(다리당 Σ|Δw|) × 2 = Table 13 131.2%(양 다리)",
        abs(2 * LIT_TURNOVER_SIGMA - LIT_T13_BASE) < 1e-9)
    chk("원문 다리당 **편도** 회전율 = 65.6/2 = 32.8% (저장소 `turn` 과 같은 척도)",
        abs(LIT_TURNOVER - LIT_TURNOVER_SIGMA / 2) < 1e-9)
    # **보조 근거다.** 원문 Table 13 의 모멘텀은 "(quintile) momentum factor" 이고
    # **보유기간이 6개월이라는 서술이 원문에 없다.** 월 재정렬이면 1/6 은 옳은 비교
    # 대상이 아니다. **척도 확정은 Panel A·C 만으로 충분**하며 이 항목은 정황일 뿐이다.
    chk("(보조) 모멘텀 편도 18.8%/월 — J-T 6개월 보유 가정 시 1/6 과 정합. "
        "**원문에 보유기간 서술 없음 → 척도 근거로 쓰지 않는다**",
        abs(LIT_MOM_SIGMA / 2 - 1 / 6) < 0.03)

    # 임계 — 가족 2칸 양측 Bonferroni
    chk("K_CRIT ≈ 2.2414", abs(K_CRIT - 2.2414) < 5e-4)

    # 원문 사양 고정값
    chk("BETA_WIN = 12 (식 5)", BETA_WIN == 12)
    chk("래그 11종 · 원문 목록과 일치",
        LAGS == (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000))
    chk("오분위(5) — 십분위가 아니다", N_QUINT == 5)
    chk("문헌 방향 +1 (양수 유의면 채택)", LIT_DIR == +1)
    # **번인은 별도 상수로 자르지 않고 `eligible()` 이 래그 11종 전부 유효를
    # 요구하는 것으로 구현된다.** 종전 검사는 `BURN_DAYS == max(LAGS)` 라
    # **정의상 참인 항진명제**였다(게이트 3차 지적). **실제 동작을 검사한다** —
    # 직전 999봉만 유효한 종목은 최장 래그(1,000)가 NaN 이라 적격에서 빠져야 한다.
    chk("번인 = 원문 §2.4 의 1,000일", BURN_DAYS == 1000)
    _n = 1200
    _X = np.full((_n, 2), 10.0)
    _V = np.ones((_n, 2), dtype=bool)
    _V[:_n - 999, 1] = False          # 1번 종목은 직전 999봉만 유효
    _X[:_n - 999, 1] = 0.0
    _row = np.array([_n - 1])
    _a0 = _rolling_mean_at(_row, _X, _V, 1000)[0, 0]
    _a1 = _rolling_mean_at(_row, _X, _V, 1000)[0, 1]
    chk("직전 1,000봉 전부 유효하면 최장 래그 산출됨", np.isfinite(_a0))
    chk("직전 999봉만 유효하면 최장 래그가 NaN → 적격에서 탈락(번인 실동작)",
        not np.isfinite(_a1))
    chk("검정력 입력은 Table 10 대응값 1.78 (헤드라인 1.63 아님)",
        LIT_SPREAD == 1.78 and LIT_HEADLINE == 1.63)
    chk("롱온리 e* = 0.80 (Table 8 유도 중앙값 0.801)",
        abs(LIT_LONGONLY - 0.80) < 1e-9)
    # Table 8 유도 산식을 **코드로 재계산**해 문서 산식과 대조한다 (게이트 2차 지적)
    q1, q2, q4, q5 = 0.31, 0.31 + 0.56, 1.93 - 0.53, 1.93
    lo_mean = (q1 + q2 + q4 + q5 + q2) / 5.0        # Q3 하한 = Q2
    hi_mean = (q1 + q2 + q4 + q5 + q4) / 5.0        # Q3 상한 = Q4
    chk("Table 8 유니버스 평균 구간 = (1.076, 1.182)",
        abs(lo_mean - 1.076) < 5e-4 and abs(hi_mean - 1.182) < 5e-4)
    chk("Q5 − 평균 구간 = (0.748, 0.854) · 중앙값 0.801",
        abs((q5 - hi_mean) - 0.748) < 5e-4 and abs((q5 - lo_mean) - 0.854) < 5e-4
        and abs((2 * q5 - lo_mean - hi_mean) / 2 - 0.801) < 5e-4)

    # 견고성 축이 **실제로 구현돼 있는가** (사전등록한 것을 돌릴 수 있어야 한다)
    import inspect
    sig = inspect.signature(run_arm).parameters
    esig = inspect.signature(eligible).parameters
    chk("§3.4 견고성 축 exec_lag · ret_cap 이 run_arm 에 구현됨",
        "exec_lag" in sig and "ret_cap" in sig)
    chk("§3.4 근사 사이즈 필터 cap_n 이 eligible · run_arm 에 구현됨",
        "cap_n" in esig and "cap_n" in sig)
    # **기능 검사** — 시그니처 존재만 보면 부족하다(게이트 3차 지적. ret_cap 에
    # 대해 같은 지적을 받아 실호출 검사를 만든 것과 같은 형태).
    _m = 3
    _P = {"market": "us",
          "A": np.ones((_m + 1, 50, len(LAGS))),
          "pm": np.full((_m + 1, 50), 100.0),
          "pit_dv": np.tile(np.arange(50, dtype=float) * 1e6 + 5e6, (_m + 1, 1))}
    _all = int(eligible(_P, 0).sum())
    _cap = int(eligible(_P, 0, cap_n=20).sum())
    chk("cap_n 없으면 적격 50종목 전부", _all == 50)
    chk("cap_n=20 이면 적격이 20종목 이하로 줄어든다", _cap <= 20 and _cap == 20)
    _dv = _P["pit_dv"][0]
    _sel = np.flatnonzero(eligible(_P, 0, cap_n=20))
    chk("cap_n 이 남기는 것은 **거래대금 상위** 20종목",
        set(_sel) == set(np.argsort(_dv)[::-1][:20]))
    # ret_cap 을 **`_hold_return` 을 실제로 호출해** 검사한다 (게이트 2차 지적 —
    # 종전에는 np.minimum 만 봐서 라벨과 검사 내용이 달랐다).
    _P = {"mret": np.array([[np.nan, np.nan], [2.5, 0.10]]), "me": np.array([0, 1]),
          "n_days": 2, "close_tr": np.ones((2, 2))}
    chk("ret_cap 미적용 시 평균 = (2.5+0.10)/2 = 1.30",
        np.isclose(_hold_return(_P, 0, np.array([0, 1])), 1.30))
    chk("ret_cap=1.0 적용 시 평균 = (1.0+0.10)/2 = 0.55",
        np.isclose(_hold_return(_P, 0, np.array([0, 1]), ret_cap=RET_CAP), 0.55))

    # E[β] 음수 슬라이스 가드 — m < 11 이면 예측을 내지 않는다
    src = inspect.getsource(forecast)
    chk("m < BETA_WIN−1 가드 존재 (미래 β 혼입 방지)",
        "m < BETA_WIN - 1" in src)

    # ── 아래는 **게이트 통과 후 작성한 `--run` 전용** 보조 함수 검증이다.
    #    사전등록 §5 의 44항목은 위까지이며, 여기부터는 판정 코드의 자기검증이다.
    chk("tstat = 평균 / (s/√n) — 손계산 대조",
        np.isclose(tstat([1.0, 2.0, 3.0, 4.0]),
                   2.5 / (np.std([1.0, 2, 3, 4], ddof=1) / 2.0)))
    chk("spearman 완전 단조 증가 = +1 · 반전 = −1",
        np.isclose(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)
        and np.isclose(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0))
    chk("cagr: 월 +1% 12개월 = 연 (1.01^12−1) = 12.68%",
        np.isclose(cagr(np.full(12, 1.0)), (1.01 ** 12 - 1) * 100))
    chk("cagr: 월 0% 이면 0%", np.isclose(cagr(np.zeros(24)), 0.0))
    # type_m — **몬테카를로와 대조**한다 (닫힌 형태의 적분 오류를 잡는 유일한 방법.
    # H-026 게이트 3회차가 이 자리에서 실제 오류를 잡았다).
    _rng = np.random.default_rng(20260819)
    _mc_ok = True
    for _se, _eff in ((0.39, 1.568), (0.27, 0.691), (1.0, 0.5)):
        _d = _eff / _se
        _z = _rng.normal(_d, 1.0, 400_000)
        _s = np.abs(_z) > K_CRIT
        _mc = float(np.abs(_z[_s]).mean() / _d)
        if abs(type_m(_se, _eff) - _mc) > 0.02:
            _mc_ok = False
    chk("type_m 닫힌 형태 ↔ 몬테카를로 일치 (3조건, 허용오차 0.02)", _mc_ok)
    chk("type_m: 효과가 SE 대비 매우 크면 1.0 에 수렴(과대추정 없음)",
        abs(type_m(0.01, 1.0) - 1.0) < 1e-6)
    chk("type_m: 검정력이 낮을수록 배율이 커진다",
        type_m(1.0, 0.3) > type_m(0.3, 0.3) > type_m(0.05, 0.3))
    # `_pooled_cost` — 표본가중 평균이 맞는가 (손계산 대조)
    _a1 = {"cost": np.array([[0.1, 0, 0, 0, 0.3]]), "n": 10}
    _a2 = {"cost": np.array([[0.2, 0, 0, 0, 0.4]]), "n": 30}
    chk("_pooled_cost 롱숏 = 표본가중((0.1+0.3),(0.2+0.4)) = 0.55",
        np.isclose(_pooled_cost({"a": _a1, "b": _a2}), (0.4 * 10 + 0.6 * 30) / 40))
    chk("_pooled_cost 롱온리 = Q5 다리만 = 0.375",
        np.isclose(_pooled_cost({"a": _a1, "b": _a2}, longonly=True),
                   (0.3 * 10 + 0.4 * 30) / 40))
    # `--run` 이 사전등록한 견고성 4축을 **실제로 호출하는가**
    rsrc = inspect.getsource(cmd_run)
    chk("--run 이 §3.4 네 축(exec_lag · ret_cap · cap_n 2종)을 전부 호출",
        "exec_lag=1" in rsrc and "ret_cap=RET_CAP" in rsrc
        and "cap_n=CAP_N[mk]" in rsrc and "cap_n=CAP_N2[mk]" in rsrc)
    chk("--run 이 direction=LIT_DIR 을 판정에 넘긴다 (H-028 결함 재발 방지)",
        "direction=LIT_DIR" in rsrc and "direction=LIT_DIR" in inspect.getsource(_report_cell))
    chk("--run 이 §3.3 플라시보 4종을 전부 돌린다",
        all(p in rsrc for p in ("P1 σ정렬", "P2 σ직교화", "P3 계수무작위", "P4 규모정렬")))
    chk("견고성 축 N 은 저장소 규약값(베이스라인 C 400/500)과 그 절반",
        CAP_N == {"kr": 400, "us": 500} and CAP_N2 == {"kr": 200, "us": 250})
    chk("--run 이 플라시보를 **본안 표본으로 맞춰** 크기를 비교한다 (§3.3 조건 3)",
        "s.index.isin(ser[mk].index)" in rsrc)
    # 표본 정렬 자체가 의도대로 되는가 — 합성 인덱스로 실동작 확인
    _ix = pd.period_range("2020-01", periods=10, freq="M")
    _s = pd.Series(np.arange(10.0), index=_ix)
    _base = pd.Series(np.zeros(4), index=_ix[3:7])
    chk("표본 정렬: 본안 인덱스에 있는 달만 남는다",
        list(_s[_s.index.isin(_base.index)].index) == list(_ix[3:7]))

    for num, name, good in ok:
        print(f"  {'PASS' if good else 'FAIL'}  {num}. {name}")
    bad = [n for n, _, g in ok if not g]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--power"
    if arg == "--selftest":
        sys.exit(cmd_selftest())
    elif arg == "--power":
        cmd_power()
    elif arg == "--run":
        cmd_run()
    else:
        print(__doc__)
        sys.exit(2)
