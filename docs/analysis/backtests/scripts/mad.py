"""H-030 — MAD (moving average distance). **Avramov·Kaplanski·Subrahmanyam 원문 형태 그대로.**

확보 판본: SSRN `ssrn_id3567010` (최종 개정 2020-09-01, 40쪽).
표본 1977-07~2018-12 · 1,353,679 종목-월 · 13,828 기업.

원문 2단계 정의 (§1, 본문 246-300행):

    MRAT_{j,t} = MA(21)_{j,t} / MA(200)_{j,t}                       … 식 (1)
    σ_t        = 그달 MRAT 의 **횡단면 표준편차**
    Top    = MRAT 십분위 **최상** ∩ { MRAT >  1 + σ_t }
    Bottom = MRAT 십분위 **최하** ∩ { MRAT <  1 − σ_t }
    MAD = +1 (Top) · −1 (Bottom) · 0 (그 외)

  · **십분위 ∩ σ 조건**이다. 단순 십분위가 아니다 (registry 등재 키 오기 — §1.4).
  · 월말 재정렬 · 1개월 보유 · Top−Bottom zero-investment.
  · 레짐 필터 없음(원문에 없음).
  · 원문은 **값가중 전용** (648행 `Throughout the analysis, we only use
    value-weighted portfolios`) — 우리는 PIT 시총이 없어 **동일가중**. §2 이탈 #1.

원문 Table 3 Panel A (VW · 전 표본 무조건부 정렬):

    Top 1.81 · Bottom 0.23 · **Diff. 1.58\\*\\*** (%p/월)   ← 산술 검산 1.81−0.23=1.58 ✓

판정은 PROTOCOL §3 개정판(2026-08-12) — **6종 추정량이 갈리면 측정 불가**.

  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/mad.py --selftest | --power

**--power 는 SE·회전율·표본·포트폴리오 크기만 출력한다. 평균·t·부호는 출력 경로가
없다** (PROTOCOL §3.1-2 — 점추정치를 보면 사전등록이 무효가 된다).
`--run` 은 **게이트 통과 후에** 추가한다.
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
MA_SHORT = 21           # 원문 식 (1) 분자
MA_LONG = 200           # 원문 식 (1) 분모
N_DEC = 10              # 십분위
SIGMA_K = 1.0           # 원문 본문 `one plus a constant sigma` — 1σ. (2σ 는 원문 각주)

# ── 원문 인용 수치 — **전부 확보 판본에서 문자열 검색으로 확인** (PROTOCOL §3.0-1) ──
LIT_TOP = 1.81          # Table 3 Panel A · Rt+1 · VW
LIT_BOTTOM = 0.23       # 〃
LIT_SPREAD = 1.58       # 〃 Diff.  ← 검정력·Type M 의 입력 (**VW 값** — §2 이탈 #1)
LIT_DIR = +1            # 문헌이 예측하는 부호

# **규모·유동성 정합 e\* (Appendix C)** — 우리 유니버스는 유동성 상위 400/500 이라
# 전 표본(1.58)보다 이 칸들이 가깝다. **판정 임계가 아니라 병기용**이다.
#   ME Top 30%(대형)      : MAD First 1.34 · MAD Last 0.82
#   ILLIQ Bottom 30%(유동): MAD First 1.46 · MAD Last 1.20
LIT_LARGE = (1.34, 0.82)
LIT_LIQUID = (1.46, 1.20)

# 롱온리 부수 판정용 — **원문 미보고**. Figure 1(본문 709-712행)에서 유도한다:
#   1977-07 말 $1 → Top decile $960 · CRSP VW 시장 $92.
#   보유 개월 = 1977-07 말 → 2018-12 말 = (2018−1977)×12 + (12−7) = **497** 개월.
#   기하평균 월수익 차 = (960^(1/497) − 92^(1/497)) × 100
# 산출은 `lit_longonly()` 가 하고 **상수를 손으로 적지 않는다**(검산 가능하도록).
# ⚠️ 497 은 **원문이 인쇄한 값이 아니라 표본 기간에서 유도한 값**이다(원문 히트 0회).
LIT_FIG1_TOP, LIT_FIG1_MKT = 960.0, 92.0
LIT_FIG1_Y0, LIT_FIG1_M0 = 1977, 7          # 표본 시작 (본문 237-239행)
LIT_FIG1_Y1, LIT_FIG1_M1 = 2018, 12         # 표본 종료
LIT_FIG1_N = (LIT_FIG1_Y1 - LIT_FIG1_Y0) * 12 + (LIT_FIG1_M1 - LIT_FIG1_M0)

# 원문 회전율 (Table 5 Panel A) — **다리당 편도**임을 BETC 항등식으로 **판별**한다.
#   원문 BETC 식(865행): [R_lt − TO_lt×TC] − [R_st + TO_st×TC]
#   ⇒ 월 알파 ≈ (TO_l + TO_s) × TC  이므로  TC ≈ 알파/(TO_l+TO_s)
#
#   ⚠️ **이것은 근사다 — "일치"가 아니다.** 원문 866-868행이 `scaled **each month** by`
#   라고 명시하므로 원문은 **월별 시변 TO** 를 곱해 회귀한다. 아래는 평균 TO 로 나눈
#   근사이고 인쇄값과 0.8~1.7% 어긋난다. **판별력은 그래도 충분하다** — 왕복 척도로
#   읽으면 절반(47/47/51 bps)이 되어 2배 어긋나기 때문이다.
#
#   (Table 4 연 알파 ÷ 12) ÷ 0.734 :
#     8.35 → 94.80 bps  (인쇄 94 · 오차 +0.9%)     ← 편도 척도와 정합
#     8.30 → 94.23 bps  (인쇄 93 · 오차 +1.3%)
#     9.05 → 102.75 bps (인쇄 101 · 오차 +1.7%)
#     왕복 척도라면 위 값의 1/2 = 47/47/51 bps → 인쇄값의 절반. **배제된다.**
#   ⇒ TO 는 **다리당 편도**이고 TC 는 **편도 요율**. 저장소 `turn`·`COST` 와 같은 척도다.
#   (H-029 가 2배 오독한 축이며, 이 가설에서는 척도 혼동이 없다.)
#
#   교차 확인 — 원문 838행 `[These figures exclude the first month of the sample,
#   in which the turnover, by definition, is 100%.]` 로 원문 TO 는 `Σ max(0, Δw)` 이고,
#   `Σw = 1` 인 달에는 `Σ|Δw| / 2` 와 **항등**이다. 코드 `tt = ct/cost/2` 가 같은 척도다.
#   단 **첫 달은 원문이 제외하고 우리는 포함**한다(§8-12).
LIT_TURN_TOP = 0.349
LIT_TURN_BOT = 0.385
LIT_BETC = 0.94         # %/월 · 편도 요율 · 3FF+UMD 1개월 보유

# ── 저장소 규약 (PROTOCOL §2 · 실행 위생이라 §0.3 에서 유지) ────────────────
PRICE_FLOOR = {"kr": 1000.0, "us": 5.0}    # 원문 $5 그대로 · KR 은 저장소 대응물
DV_WIN = 20
DV_MIN = {"kr": None, "us": 2e6}           # PROTOCOL §2 (PIT) · KR 은 하한 없음
COST = {"kr": 0.0014, "us": 0.0005}        # **편도** — 왕복(0.28%/0.10%)의 절반
RET_CAP = 1.0                              # 종목-월 수익 상한 (견고성 축)
SMA_BENCH = 200                            # 레짐은 원문에 없다 — 보고용으로만

# 다중비교 가족 — **결과를 보기 전에 고정**.
#   ① 롱숏 Top−Bottom (주 판정)   ② 롱온리 Top−유니버스 (부수 판정)
FAMILY = 2
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414

SEED = 20260819                            # 플라시보 P4 난수 시드

# 근사 사이즈 필터 견고성 축 (PROTOCOL §1.1-2). 저장소 기존 규약값 그대로.
CAP_N = {"kr": 400, "us": 500}
CAP_N2 = {"kr": 200, "us": 250}

# 리밸일 우연의 CAGR 표준편차 — H-016 실측. 절대 성과 보고용(차분에는 해당 없음).
TIMING_LUCK_SD = {"kr": 4.02, "us": 1.43}

MOM_SKIP, MOM_WIN = 1, 11                  # 플라시보 P2 — 원문 MOM = 월 −12~−2
H52_WIN = 252                              # 플라시보 P3 — 원문 52H
SIG_WIN, SIG_MIN = 250, 200                # 플라시보 P1 — 250일 일간수익률 σ

# `sigma_cs` 의 ddof 선택이 σ_t 를 바꾸는 최대 크기 = 1 − √((n−1)/n) · n = 최소 적격 수.
# 실측 최소 적격은 KR 409 · US 1,121 → 최대 **0.1223%**. selftest 가 실데이터로 확인한다.
SIGMA_DDOF_TOL = 0.13                      # %


def lit_longonly() -> float:
    """Figure 1 에서 유도한 롱온리 e\* (%p/월, 기하). **원문 미보고 값이다.**"""
    g = lambda v: v ** (1.0 / LIT_FIG1_N) - 1.0
    return (g(LIT_FIG1_TOP) - g(LIT_FIG1_MKT)) * 100.0


# ────────────────────────────────────────────────────────────── 패널

def _roll_mean(rows: np.ndarray, X: np.ndarray, V: np.ndarray, L: int) -> np.ndarray:
    """`rows` 각 행에서 **직전 L 봉(그 행 포함)** 종가의 단순평균.

    누적합으로 O(일수×종목). `V` 는 유효 마스크이고 **L 봉이 전부 유효할 때만**
    값을 낸다(부분 창을 평균하지 않는다).
    """
    S = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    K = np.concatenate([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)], axis=0)
    hi, lo = rows + 1, rows + 1 - L
    out = np.full((len(rows), X.shape[1]), np.nan)
    ok = lo >= 0
    if not ok.any():
        return out
    h, l = hi[ok], lo[ok]
    cnt = K[h] - K[l]
    out[ok] = np.where(cnt == L, (S[h] - S[l]) / L, np.nan)
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

    # **신호용 종가** — `close > 0` 만 본다. KR 거래정지 봉은 open=high=low=0 이고
    # close 만 직전가로 이월돼 있다(PROTOCOL §1). CRSP 도 무거래일에 직전가를 남기므로
    # 이월 종가를 이동평균에 쓰는 것이 원문에 가장 가깝다.
    sig_ok = np.isfinite(cl_raw) & (cl_raw > 0)
    Xs = np.where(sig_ok, cl_raw, 0.0)

    # **체결·수익률용 종가** — 거래 가능한 봉만 (PROTOCOL §1: open<=0 도 마스킹)
    trade_ok = sig_ok & np.isfinite(op_raw) & (op_raw > 0)
    cl_tr = np.where(trade_ok, cl_raw, np.nan)

    mk = pd.PeriodIndex(pd.to_datetime(idx), freq="M")
    me = pd.Series(np.arange(len(idx)), index=mk).groupby(level=0).last().to_numpy()
    me = me[:-1]                                  # 마지막 역월은 부분 월 → 제외
    months = pd.PeriodIndex(mk[me])

    pm = cl_tr[me]                                # 월말 종가 (거래 가능한 봉만)
    mret = np.full_like(pm, np.nan)
    mret[1:] = pm[1:] / pm[:-1] - 1.0             # 월수익 (월말→월말)

    # MRAT = MA(21)/MA(200) — 원문 식 (1)
    ma_s = _roll_mean(me, Xs, sig_ok, MA_SHORT)
    ma_l = _roll_mean(me, Xs, sig_ok, MA_LONG)
    with np.errstate(all="ignore"):
        mrat = ma_s / ma_l
    mrat[~np.isfinite(mrat)] = np.nan

    dv_d = piv["close"] * piv["volume"]
    pit_dv = dv_d.rolling(DV_WIN, min_periods=DV_WIN).mean().to_numpy(float)[me]

    # ── 플라시보 축 (전부 월말 기준 · look-ahead 없음) ──
    cl_sig = np.where(sig_ok, cl_raw, np.nan)
    dret = np.full_like(cl_raw, np.nan)
    with np.errstate(all="ignore"):
        dret[1:] = cl_sig[1:] / cl_sig[:-1] - 1.0
    dret[~np.isfinite(dret)] = np.nan
    sig = pd.DataFrame(dret).rolling(SIG_WIN, min_periods=SIG_MIN).std().to_numpy(float)[me]

    # MOM = 월 −12~−2 수익 (원문 Appendix A). 월말 인덱스 기준 pm[m-12]→pm[m-1]
    mom = np.full_like(pm, np.nan)
    if len(pm) > MOM_SKIP + MOM_WIN:
        s = MOM_SKIP + MOM_WIN
        with np.errstate(all="ignore"):
            mom[s:] = pm[s - MOM_SKIP: len(pm) - MOM_SKIP] / pm[:len(pm) - s] - 1.0

    # 52H = 월말 종가 / 직전 252봉 **종가** 최고 (원문 Appendix A 1791행:
    # `current price/highest price during the last 52 weeks`).
    # **고가가 아니라 종가다** — 원문 Table 1 의 52H 최댓값·95분위가 둘 다 1.000 인 것이
    # 종가 기준임을 확정한다(고가 기준이면 1 에 거의 닿지 않는다). 게이트 1차 지적 5.
    roll_hi = pd.DataFrame(cl_sig).rolling(H52_WIN, min_periods=H52_WIN).max().to_numpy(float)[me]
    pm_sig = np.where(sig_ok[me], cl_raw[me], np.nan)
    with np.errstate(all="ignore"):
        h52 = pm_sig / roll_hi
    h52[~np.isfinite(h52)] = np.nan

    b = loading.load_bench(market).set_index("date")["close"].reindex(idx).ffill()
    bv = b.to_numpy(float)
    sma = pd.Series(bv).rolling(SMA_BENCH, min_periods=SMA_BENCH).mean().to_numpy()

    return {"tickers": np.asarray(cols, dtype=str), "months": months, "me": me,
            "mrat": mrat, "pm": pm, "mret": mret, "pit_dv": pit_dv, "sigma": sig,
            "mom": mom, "h52": h52, "close_tr": cl_tr, "n_days": len(idx),
            "bull": (bv[me] > sma[me]), "market": market}


# ──────────────────────────────────────────────────────── 적격 · 포트폴리오

def eligible(P: dict, m: int, *, cap_n: int | None = None) -> np.ndarray:
    """월 m 말 기준 적격 — MRAT 유효 + 가격 하한 + PIT 거래대금.

    `cap_n` 은 PROTOCOL §1.1-2 **근사 사이즈 필터** 견고성 축 — PIT 거래대금 상위 N.
    과거 시점 시총이 없어 거래대금을 규모 대리변수로 쓴다. **주 판정은 `cap_n=None`.**
    """
    mkt = P["market"]
    ok = np.isfinite(P["mrat"][m])
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


def sigma_cs(P: dict, m: int, ok: np.ndarray) -> float:
    """σ_t = **그달 적격 종목 MRAT 의 횡단면 표준편차** (원문 Table 3·4 캡션).

    표본표준편차(ddof=1)를 쓴다 — 원문은 ddof 를 명시하지 않는다. ddof 선택이 σ_t 를
    바꾸는 크기는 `1 − √((n−1)/n)` 이고 실측 최소 적격(KR 409 · US 1,121)에서
    **최대 0.1223%** 다. selftest 가 **실데이터의 최소 적격 수로** 확인한다.
    """
    v = P["mrat"][m][ok]
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def mad_legs(score: np.ndarray, ok: np.ndarray, tickers: np.ndarray,
             sig: float, *, k: float = SIGMA_K) -> tuple:
    """원문 2단계 — 십분위 → σ 조건. 반환 (top, bottom, n_d10, n_d1).

    **정렬은 오름차순, 동점은 티커 오름차순**(재현성). `np.array_split` 규약대로
    나머지는 앞쪽(=최하 십분위) 그룹에 붙는다.
    `top`/`bottom` 은 σ 조건 **적용 후**, `n_d10`/`n_d1` 은 **적용 전** 십분위 크기다.
    """
    cand = np.flatnonzero(ok & np.isfinite(score))
    if len(cand) < N_DEC * 2 or not np.isfinite(sig):
        return np.array([], int), np.array([], int), 0, 0
    order = np.array(sorted(cand, key=lambda j: (score[j], tickers[j])), dtype=int)
    groups = np.array_split(order, N_DEC)
    d1, d10 = np.asarray(groups[0], int), np.asarray(groups[-1], int)
    top = d10[score[d10] > 1.0 + k * sig]
    bot = d1[score[d1] < 1.0 - k * sig]
    return top, bot, len(d10), len(d1)


def _hold_return(P: dict, m: int, mem: np.ndarray, *, exec_lag: int = 0,
                 ret_cap: float | None = None) -> float:
    """월 m 말 편입 → 월 m+1 **동일가중** 수익률. 원문은 월말 종가 체결(`exec_lag=0`).

    보유 종목 중 다음 달 수익이 NaN 인 종목은 평균에서 탈락시킨다(상장폐지·장기정지).
    """
    if len(mem) == 0:
        return float("nan")
    if exec_lag == 0:
        r = P["mret"][m + 1][mem]
    else:
        i0, i1 = P["me"][m] + exec_lag, P["me"][m + 1] + exec_lag
        if i1 >= P["n_days"]:
            return float("nan")
        c = P["close_tr"]
        with np.errstate(all="ignore"):
            r = c[i1][mem] / c[i0][mem] - 1.0
    r = r[np.isfinite(r)]
    if ret_cap is not None:
        r = np.minimum(r, ret_cap)
    return float(r.mean()) if len(r) else float("nan")


def _turn_cost(prev, mem, drift, cost):
    """직전 보유가 수익률로 표류한 뒤의 비중 대비 회전율 × 비용. 반환 (비용, 새 비중)."""
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
    # `turn` 은 Σ|Δw| 라 전량 교체 시 2.0 이다. 편도 회전율은 그 절반.
    return cost * turn, w_new


def run_arm(P: dict, *, score: np.ndarray | None = None, exec_lag: int = 0,
            ret_cap: float | None = None, cap_n: int | None = None,
            k: float = SIGMA_K) -> dict:
    """MAD 팔을 돌린다. `score=None` 이면 본안(MRAT).

    **어느 한 다리라도 비면 그 달을 표본에서 제외**하고 `skipped` 로 센다 —
    원문은 빈 다리를 다루지 않으므로 사전등록에서 고정한 규칙이다(§3.5).
    """
    n_m = len(P["months"])
    cost = COST[P["market"]]
    sc_all = P["mrat"] if score is None else score
    # σ 조건은 **MRAT 기준**이다. 플라시보는 정렬축만 바꾸고 σ 조건은 그대로 둔다
    # (조건을 함께 바꾸면 무엇이 달라졌는지 분리되지 않는다 — 문서 §3.6).
    prev_t = prev_b = None
    rows = {k_: [] for k_ in ("top", "bot", "uni", "ct", "cb", "tt", "tb",
                              "nt", "nb", "nd", "el", "sg")}
    months = []
    # **제외 사유를 분리해 센다** (게이트 1차 지적 — 문서가 전부 "빈 다리"로 적었다)
    sk = {"적격부족": 0, "빈다리": 0, "수익불가": 0}
    for m in range(n_m - 1):
        ok = eligible(P, m, cap_n=cap_n)
        if ok.sum() < N_DEC * 2:
            sk["적격부족"] += 1
            continue
        sig = sigma_cs(P, m, ok)
        top, bot, nd10, nd1 = mad_legs(sc_all[m], ok, P["tickers"], sig, k=k)
        if len(top) == 0 or len(bot) == 0:
            sk["빈다리"] += 1
            continue
        drift = np.nan_to_num(P["mret"][m], nan=0.0)
        rt = _hold_return(P, m, top, exec_lag=exec_lag, ret_cap=ret_cap)
        rb = _hold_return(P, m, bot, exec_lag=exec_lag, ret_cap=ret_cap)
        ru = _hold_return(P, m, np.flatnonzero(ok), exec_lag=exec_lag, ret_cap=ret_cap)
        if not (np.isfinite(rt) and np.isfinite(rb) and np.isfinite(ru)):
            sk["수익불가"] += 1
            continue
        ct, prev_t = _turn_cost(prev_t, top, drift, cost)
        cb, prev_b = _turn_cost(prev_b, bot, drift, cost)
        rows["top"].append(rt * 100.0); rows["bot"].append(rb * 100.0)
        rows["uni"].append(ru * 100.0)
        rows["ct"].append(ct * 100.0); rows["cb"].append(cb * 100.0)
        rows["tt"].append(ct / cost / 2.0); rows["tb"].append(cb / cost / 2.0)
        rows["nt"].append(len(top)); rows["nb"].append(len(bot))
        rows["nd"].append((nd10 + nd1) / 2.0); rows["el"].append(int(ok.sum()))
        rows["sg"].append(sig)
        months.append(m)
    if not months:
        return {}
    a = {k_: np.asarray(v, float) for k_, v in rows.items()}
    spread = (a["top"] - a["bot"]) - (a["ct"] + a["cb"])
    longonly = (a["top"] - a["ct"]) - a["uni"]
    return {"spread": spread, "longonly": longonly, "months": np.asarray(months),
            "n": len(months), "skip": sk, "skipped": sum(sk.values()), **a}


# ────────────────────────────────────────────────────────────── 플라시보

def placebo_scores(P: dict) -> dict:
    """문서 §3.6 사전 배치 플라시보 — **결과를 보기 전에 고정했다.**

    P1 σ 정렬    : 250일 일간수익률 표준편차. H-028·H-033 을 연속으로 죽인 축
    P2 MOM 정렬  : 월 −12~−2 수익. **원문 Table 2·3 의 최대 경쟁 변수**(ρ(MRAT,MOM)=+0.584)
    P3 52H 정렬  : 종가/직전 252봉 최고가. **원문의 두 번째 경쟁 변수**(ρ=+0.636)
    P4 MRAT 무작위: MRAT 을 **시드 고정 무작위 순열**로 섞는다(신호 파괴 · 기계 검증)

    **σ 조건은 넷 다 MRAT 기준 그대로 둔다** — 정렬축만 바꿔야 무엇이 달라졌는지
    분리된다. 그래서 P1~P3 은 "십분위를 다른 축으로 뽑되 σ 게이트는 동일"이다.
    """
    rng = np.random.default_rng(SEED)
    p4 = np.full_like(P["mrat"], np.nan)
    for m in range(P["mrat"].shape[0]):
        v = P["mrat"][m]
        fin = np.flatnonzero(np.isfinite(v))
        if len(fin):
            p4[m, fin] = v[rng.permutation(fin)]
    return {"P1 σ정렬": P["sigma"], "P2 MOM정렬": P["mom"],
            "P3 52H정렬": P["h52"], "P4 MRAT무작위": p4}


# ─────────────────────────────────────────────────────────────── 명령

def _pc():
    import pooled_clustering as PC
    return PC


_PANELS: dict = {}


def _panels() -> dict:
    if not _PANELS:
        for mk in ("kr", "us"):
            _PANELS[mk] = build_panel(mk)
    return _PANELS


def cmd_power():
    PC = _pc()
    lo_lit = lit_longonly()
    print("=" * 100)
    print("[H-030] 사전 검출력 — **SE·회전율·표본·포트폴리오 크기만**.")
    print("        평균·t·부호는 출력 경로가 없다 (PROTOCOL §3.1-2).")
    print(f"  가족 {FAMILY}칸 Bonferroni → |t| > {K_CRIT:.4f}")
    print(f"  ① e* = {LIT_SPREAD}%p/월 (Table 3 Panel A Diff. · **VW**)")
    print(f"  ② e* = {lo_lit:.4f}%p/월 (Figure 1 에서 유도 · **원문 미보고**)")
    print(f"  원문 회전율 Top {LIT_TURN_TOP*100:.1f}% · Bottom {LIT_TURN_BOT*100:.1f}%"
          f" (다리당 편도 — BETC 근사가 왕복 척도를 배제)")
    print("  검정력은 PROTOCOL §3 개정판대로 **6종 중 SE 최대(= 최저 검정력)** 기준.")
    print("=" * 100)
    ser, lo_ser = {}, {}
    cst_n = cst_w = lo_n = lo_w = 0.0
    for mk, P in _panels().items():
        a = run_arm(P)
        if not a:
            print(f"\n── {mk.upper()} ── 산출 불가")
            continue
        mm = P["months"][a["months"]]
        ser[mk] = pd.Series(a["spread"], index=mm)
        lo_ser[mk] = pd.Series(a["longonly"], index=mm)
        cm = float(a["ct"].mean() + a["cb"].mean())
        cst_n += cm * a["n"]; cst_w += a["n"]
        lo_n += float(a["ct"].mean()) * a["n"]; lo_w += a["n"]
        sk = " · ".join(f"{k} {v}" for k, v in a["skip"].items())
        print(f"\n── {mk.upper()} ── n={a['n']}개월 ({mm[0]}~{mm[-1]})"
              f" · 제외 {a['skipped']}개월 ({sk})")
        print(f"   적격 평균 {a['el'].mean():.0f}종목 · **최소 {int(a['el'].min())}**"
              f" (→ ddof 영향 최대 {(1-np.sqrt((a['el'].min()-1)/a['el'].min()))*100:.4f}%)")
        print(f"   십분위 평균 {a['nd'].mean():.1f}"
              f" → σ 조건 후 Top {a['nt'].mean():.1f} · Bottom {a['nb'].mean():.1f}"
              f" (통과율 {(a['nt'].mean()+a['nb'].mean())/2/a['nd'].mean()*100:.1f}%)")
        for leg, nm in (("nt", "Top"), ("nb", "Bottom")):
            v = a[leg]
            print(f"   {nm:<6} 최소 {int(v.min())} · 5%분위 {np.percentile(v,5):.0f}"
                  f" · 중위 {np.median(v):.0f} · 최대 {int(v.max())}"
                  f" — **5종목 미만 {int((v<5).sum())}개월 · 10종목 미만 {int((v<10).sum())}개월**")
        print(f"   σ_t(횡단면 MRAT 표준편차) 중위 {np.median(a['sg']):.4f}"
              f" · 범위 {a['sg'].min():.4f}~{a['sg'].max():.4f}")
        print(f"   회전율(다리당 편도) Top {a['tt'].mean()*100:.1f}%"
              f" · Bottom {a['tb'].mean()*100:.1f}%/월"
              f" — 원문의 {a['tt'].mean()/LIT_TURN_TOP:.2f}배 · {a['tb'].mean()/LIT_TURN_BOT:.2f}배")
        print(f"   스프레드 월 비용 {cm:.4f}%p → e*({LIT_SPREAD})의 {cm/LIT_SPREAD*100:.1f}% 잠식")
        print(f"   naive SE {PC.naive_se({'x': ser[mk]})[1]:.4f}%p (참고)")
    if len(ser) < 2:
        print("\n두 시장 모두 산출되지 않아 통합 검정력을 낼 수 없다.")
        return
    cells = (("① 롱숏 Top−Bottom", ser, cst_n / max(cst_w, 1), LIT_SPREAD,
              "Table 3 Panel A Diff. (VW · 전 표본)"),
             ("② 롱온리 Top−유니버스", lo_ser, lo_n / max(lo_w, 1), lo_lit,
              "Figure 1 에서 유도 — **원문 미보고**"))
    for tag, s, cst, lit, src in cells:
        est = PC.all_estimates(s)
        ses = {k: est[k] for k in PC.VOTERS}
        worst = max(ses, key=lambda k: ses[k])
        net = lit - cst
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
        print(f"   e* {lit:.4f}%p/월 — {src}")
        print(f"   표본가중 비용 {cst:.4f}%p/월 → net {net:.4f}%p")
        print(f"   **SE 최대 = {worst} ({est[worst]:.4f}) → 검정력(비용 후) = {pw*100:.1f}%**"
              f"  (gross 였다면 {pwg*100:.1f}%)")
        if tag.startswith("①"):
            print("   규모·유동성 정합 e* 로 본 검정력 (병기 — 임계 아님):")
            for nm, pair in (("ME Top 30%", LIT_LARGE), ("ILLIQ Bottom 30%", LIT_LIQUID)):
                for lab, v in zip(("First", "Last"), pair):
                    dd = (v - cst) / est[worst]
                    p2 = _N.cdf(dd - K_CRIT) + _N.cdf(-dd - K_CRIT)
                    print(f"      {nm} MAD {lab}: e*={v} → {p2*100:.1f}%")


# ─────────────────────────────────────────────────────────────── selftest

def cmd_selftest():
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print(f"  {'✓' if cond else '✗'} {name}")

    print("=" * 100)
    print("[H-030] selftest")
    print("=" * 100)

    # ── 원문 인용 산술 ──
    print("\n[원문 산술]")
    chk("Table 3 Panel A 검산 1.81 − 0.23 = 1.58",
        abs((LIT_TOP - LIT_BOTTOM) - LIT_SPREAD) < 1e-9)
    # ⚠️ **근사다.** 원문은 월별 시변 TO 를 곱해 회귀하므로 평균 TO 근사와 0.8~1.7% 어긋난다.
    chk("BETC 근사 8.35/12 ÷ 0.734 = 94.80bps — 인쇄값 94 와 **2% 이내**",
        abs((8.35 / 12) / (LIT_TURN_TOP + LIT_TURN_BOT) * 100 / 94.0 - 1) < 0.02)
    chk("BETC 근사 9.05/12 ÷ 0.734 = 102.75bps — 인쇄값 101 과 **2% 이내**",
        abs((9.05 / 12) / (LIT_TURN_TOP + LIT_TURN_BOT) * 100 / 101.0 - 1) < 0.02)
    chk("BETC 근사가 **왕복 척도를 배제**한다 (왕복이면 인쇄값의 절반)",
        abs((8.35 / 12) / (LIT_TURN_TOP + LIT_TURN_BOT) / 2 * 100 - 47.4) < 0.5)
    # 보유 개월 — **독립 산술**로 확인한다. 상수를 그대로 되풀이하지 않는다.
    chk("Figure 1 보유 개월 = 497 (1977-07 말 → 2018-12 말)", LIT_FIG1_N == 497)
    chk("보유 개월이 연 단위 검산과 일치 (41년 5개월 = 41×12+5)",
        LIT_FIG1_N == 41 * 12 + 5)
    # **항진명제 금지** — 좌변은 `lit_longonly()`, 우변은 손으로 계산한 독립 상수다.
    #   960^(1/497) = 1.0139131…  →  1.39131%/월
    #    92^(1/497) = 1.0091401…  →  0.91401%/월
    #   차 = 0.47730%p/월
    chk("Figure 1 유도값 = 1.39131% − 0.91401% = 0.47730%p/월 (독립 계산)",
        abs(lit_longonly() - 0.47730) < 5e-5)
    chk("498 개월로 잘못 잡으면 값이 달라진다 (상수가 실제로 쓰인다는 증거)",
        abs((LIT_FIG1_TOP ** (1 / 498) - LIT_FIG1_MKT ** (1 / 498)) * 100
            - lit_longonly()) > 1e-4)
    chk("Bonferroni 임계 = 2.2414 (가족 2칸)", abs(K_CRIT - 2.2414) < 5e-5)

    # ── _roll_mean 소형 예제 ──
    print("\n[_roll_mean — 소형 수치 예제]")
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    V = np.ones_like(X, bool)
    r = _roll_mean(np.array([4]), X, V, 3)
    chk("직전 3봉 평균 (3+4+5)/3 = 4.0", abs(r[0, 0] - 4.0) < 1e-12)
    r0 = _roll_mean(np.array([1]), X, V, 3)
    chk("창이 표본을 벗어나면 NaN", not np.isfinite(r0[0, 0]))
    V2 = V.copy(); V2[3, 0] = False
    X2 = X.copy(); X2[3, 0] = 0.0
    r2 = _roll_mean(np.array([4]), X2, V2, 3)
    chk("창 안에 무효 봉이 있으면 NaN (부분 창 평균 금지)", not np.isfinite(r2[0, 0]))

    # ── mad_legs 소형 예제 ──
    print("\n[mad_legs — 소형 수치 예제]")
    sc = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.5,
                   0.6, 0.75, 0.85, 0.92, 0.97, 1.06, 1.12, 1.25, 1.35, 1.6])
    okm = np.ones(20, bool)
    tk = np.array([f"T{i:02d}" for i in range(20)])
    t, b, n10, n1 = mad_legs(sc, okm, tk, 0.10)
    chk("십분위 크기 = 20/10 = 2", n10 == 2 and n1 == 2)
    chk("최상 십분위 = {1.5, 1.6} 중 MRAT > 1.10 → 둘 다 통과",
        set(sc[t]) == {1.5, 1.6})
    chk("최하 십분위 = {0.5, 0.6} 중 MRAT < 0.90 → 둘 다 통과",
        set(sc[b]) == {0.5, 0.6})
    t2, b2, _, _ = mad_legs(sc, okm, tk, 0.60)          # 1+σ = 1.60 · 1−σ = 0.40
    chk("σ=0.60 → 문턱 1.60. D10 최댓값이 정확히 1.6 이라 **엄격 > 로 탈락 → 0종목**",
        len(t2) == 0)
    chk("σ=0.60 → 문턱 0.40. D1 최솟값 0.5 > 0.40 이라 **0종목**", len(b2) == 0)
    t3, b3, _, _ = mad_legs(sc, okm, tk, 0.0)
    chk("σ=0 이면 십분위 그대로 (Top 2 · Bottom 2)", len(t3) == 2 and len(b3) == 2)
    small = mad_legs(sc[:5], np.ones(5, bool), tk[:5], 0.1)
    chk("적격이 십분위×2 미만이면 빈 다리", len(small[0]) == 0 and len(small[1]) == 0)
    nan_sig = mad_legs(sc, okm, tk, float("nan"))
    chk("σ 가 NaN 이면 빈 다리", len(nan_sig[0]) == 0)

    # ── sigma_cs ──
    print("\n[sigma_cs]")
    Pt = {"mrat": np.array([[1.0, 1.2, 0.8, np.nan]])}
    s = sigma_cs(Pt, 0, np.ones(4, bool))
    chk("횡단면 표준편차 ddof=1 (NaN 제외)",
        abs(s - np.std([1.0, 1.2, 0.8], ddof=1)) < 1e-12)
    chk("유효 1개면 NaN", not np.isfinite(sigma_cs({"mrat": np.array([[1.0, np.nan]])},
                                                   0, np.ones(2, bool))))

    # ── _turn_cost ──
    print("\n[_turn_cost]")
    d = np.zeros(4)
    c1, w1 = _turn_cost(None, np.array([0, 1]), d, 0.01)
    chk("첫 달은 전량 신규 → Σ|Δw| = 1.0 → 비용 = 요율", abs(c1 - 0.01) < 1e-12)
    c2, _ = _turn_cost(w1, np.array([0, 1]), d, 0.01)
    chk("동일 보유·표류 0 → 회전율 0", abs(c2) < 1e-12)
    c3, _ = _turn_cost(w1, np.array([2, 3]), d, 0.01)
    chk("전량 교체 → Σ|Δw| = 2.0 → 비용 = 2×요율", abs(c3 - 0.02) < 1e-12)

    # ── 척도 정합 ──
    print("\n[척도 정합]")
    chk("COST 는 편도 = PROTOCOL §2 왕복의 절반 (KR)", abs(COST["kr"] - 0.28 / 100 / 2) < 1e-12)
    chk("COST 는 편도 = PROTOCOL §2 왕복의 절반 (US)", abs(COST["us"] - 0.10 / 100 / 2) < 1e-12)

    # ── 패널 무결성 (실데이터) ──
    print("\n[패널 — 실데이터]")
    P = build_panel("us")
    chk("MRAT 이 유한한 칸에서 전부 양수", np.nanmin(P["mrat"]) > 0)
    chk("월 인덱스가 단조 증가", bool((np.diff(P["me"]) > 0).all()))
    m_last = len(P["months"]) - 1
    chk("첫 월의 mret 는 전부 NaN (직전 월말이 없다)",
        not np.isfinite(P["mret"][0]).any())
    e0 = eligible(P, m_last)
    e_cap = eligible(P, m_last, cap_n=20)
    chk("cap_n 은 부분집합", bool((e_cap & e0 == e_cap).all()))
    chk("cap_n=20 이면 20종목 이하", int(e_cap.sum()) <= 20)
    dvv = P["pit_dv"][m_last]
    chk("cap_n 이 거래대금 상위를 남긴다",
        (not e_cap.any()) or np.nanmin(dvv[e_cap]) >= np.nanmax(
            dvv[e0 & ~e_cap]) - 1e-6 if (e0 & ~e_cap).any() else True)

    # ── look-ahead — **미래 행을 실제로 오염시켜 결과가 불변인지 본다** ──
    # (게이트 1차 지적: 종전 두 항목은 docstring 검색·비어있지 않음 확인이라
    #  look-ahead 를 전혀 검증하지 않았다.)
    print("\n[look-ahead — 미래 행 오염 실험]")
    mm = m_last - 3          # m+2 까지 유효해야 아래 "m+2 불변" 실험이 실제로 돈다
    okm2 = eligible(P, mm)
    sgm = sigma_cs(P, mm, okm2)
    tp, bt, _, _ = mad_legs(P["mrat"][mm], okm2, P["tickers"], sgm)
    chk("기준 실험이 성립한다 (양 다리가 비지 않음)", len(tp) > 0 and len(bt) > 0)

    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    for key in ("mrat", "pm", "pit_dv", "mret", "sigma", "mom", "h52"):
        Q[key][mm + 1:] = np.nan                     # **월 mm 이후를 전부 파괴**
    okq = eligible(Q, mm)
    sgq = sigma_cs(Q, mm, okq)
    tq, bq, _, _ = mad_legs(Q["mrat"][mm], okq, Q["tickers"], sgq)
    chk("미래 행을 전부 NaN 으로 만들어도 적격 집합이 불변", bool((okq == okm2).all()))
    chk("미래 행을 전부 NaN 으로 만들어도 σ_t 가 불변", abs(sgq - sgm) < 1e-12)
    chk("미래 행을 전부 NaN 으로 만들어도 Top/Bottom 이 불변",
        np.array_equal(tq, tp) and np.array_equal(bq, bt))

    # 보유수익은 **정확히 m+1 행**을 읽는다 — m+1 을 바꾸면 변하고 m+2 를 바꾸면 안 변한다
    base = _hold_return(P, mm, tp)
    R1 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    R1["mret"][mm + 1] = R1["mret"][mm + 1] + 0.5
    chk("mret[m+1] 을 +0.5 흔들면 보유수익이 +0.5 움직인다",
        abs(_hold_return(R1, mm, tp) - (base + 0.5)) < 1e-9)
    if mm + 2 < P["mret"].shape[0]:
        R2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
        R2["mret"][mm + 2] = R2["mret"][mm + 2] + 0.5
        chk("mret[m+2] 를 흔들어도 보유수익은 불변 (m+2 를 읽지 않는다)",
            abs(_hold_return(R2, mm, tp) - base) < 1e-12)

    # ── 플라시보 ──
    print("\n[플라시보]")
    pl = placebo_scores(P)
    chk("플라시보 4종", len(pl) == 4)
    chk("P4 는 MRAT 의 순열이라 월별 값집합이 동일",
        np.allclose(np.sort(pl["P4 MRAT무작위"][m_last][
                        np.isfinite(pl["P4 MRAT무작위"][m_last])]),
                    np.sort(P["mrat"][m_last][np.isfinite(P["mrat"][m_last])])))
    chk("P4 는 시드 고정 — 두 번 호출해도 동일",
        np.allclose(placebo_scores(P)["P4 MRAT무작위"][m_last],
                    pl["P4 MRAT무작위"][m_last], equal_nan=True))
    chk("P2 MOM 이 월 −12~−2 창 (앞 12개월은 NaN)",
        not np.isfinite(pl["P2 MOM정렬"][:MOM_SKIP + MOM_WIN]).any())
    chk("P3 52H 가 (0, 1] 범위", np.nanmax(pl["P3 52H정렬"]) <= 1.0 + 1e-9)
    chk("P3 52H 가 **종가** 기준이라 1.000 에 실제로 닿는다 (원문 Table 1 최댓값 1.000)",
        np.nanmax(pl["P3 52H정렬"]) > 1.0 - 1e-9)

    # ── ddof — **실데이터의 최소 적격 수로** 확인 (docstring 주장의 근거) ──
    print("\n[σ 의 ddof 영향 — 실데이터]")
    worst = 10 ** 9
    for mk_ in ("kr", "us"):
        aa = run_arm(build_panel(mk_))
        n_min = int(aa["el"].min())
        worst = min(worst, n_min)
        print(f"     {mk_.upper()} 최소 적격 {n_min}종목"
              f" → ddof 영향 {(1-np.sqrt((n_min-1)/n_min))*100:.4f}%")
    chk(f"ddof 선택의 영향이 {SIGMA_DDOF_TOL}% 미만 (최소 적격 {worst}종목 기준)",
        (1 - np.sqrt((worst - 1) / worst)) * 100 < SIGMA_DDOF_TOL)

    print("\n" + "=" * 100)
    print(f"selftest {ok[0]}/{ok[1]}")
    print("=" * 100)
    return ok[0] == ok[1]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if arg == "--selftest":
        sys.exit(0 if cmd_selftest() else 1)
    elif arg == "--power":
        cmd_power()
    else:
        print(__doc__)
        sys.exit(2)
