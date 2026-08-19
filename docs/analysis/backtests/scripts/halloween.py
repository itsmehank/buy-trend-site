"""H-032 — Halloween / Sell-in-May. **Jacobsen & Zhang 원문 형태 그대로.**

출처: Ben Jacobsen · Cherry Y. Zhang, "The Halloween Indicator: Everywhere and all
the time" (워킹페이퍼, SSRN id2154873 · 55,425 관측 · 108개국 · 1693~2011).
게재본은 *JIMF* 110 (2021) 102268 (62,962 관측) — **판본이 다르므로 워킹페이퍼
수치만 쓴다**(문서 §1.2).

원문 식 (1) — 본문 400·406-408행:

    r_t = α + β · D_t + ε_t

    r_t = **연속복리(로그) 월 지수수익률**
    D_t = 1 if 월 ∈ {11,12,1,2,3,4} (November~April), else 0
    β   = 겨울·여름 **두 6개월 구간의 평균수익 차**

원문 **411행**: `significantly positive, as it represents the difference between the
mean returns` for the two 6-month periods of November-April and May-October.
(400·406-408행은 식 (1)과 `r_t`·`D_t` 의 정의 행이다 — 행번호를 나눠 적는다.)

**척도 주의** — Table 3 의 Mean 열은 **월수익 × 6** 이고 β 는 그 차다. 즉
`β(6개월 %p) = 6 × (월 평균수익 차)`. 검산: KR 12.25 − 1.26 = 10.99 ≈ 11.00 ·
US 2.24 − 0.57 = 1.67. (`--selftest` 가 이 항등식을 검사한다.)

**e\\* 는 통합 108개국 4.52 가 아니다** — 그것은 108개국 통합이고 우리는 2개 시장이다.
**Table 3 국가별 값**(KR 11.00 · US 1.67)을 쓴다. 두 값이 6.6배 차이나므로
크기 불일치를 판정문에 명기한다.

**원문의 주 명세는 월별 회귀다.** 6개월(반기) 데이터는 원문 스스로 §5.3 에서
**견고성 검정**으로 돌린다. 따라서 여기서도 월별이 주 판정, 반기가 견고성이다.

판정은 PROTOCOL §3 개정판(2026-08-12) — **6종 추정량이 갈리면 측정 불가**.

  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \\
    docs/analysis/backtests/scripts/halloween.py --selftest | --power

**--power 는 SE·표본·전략 회전율만 출력한다. 평균·t·부호는 출력 경로가 없다**
(PROTOCOL §3.1-2 — 점추정치를 보면 사전등록이 무효가 된다).
`--run` 은 게이트 통과 후에 작성한다.
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
WINTER = (11, 12, 1, 2, 3, 4)          # 원문 407행 `November through April`
SEASON_LEN = 6                          # β 척도: 월 차 × 6 = Table 3 의 β

# 원문 Table 3 (국가별 전체 표본) — **사전 고정**. 단위 = **6개월 %p**
#   KR 02/1962~07/2011: Nov-Apr 12.25 · May-Oct 1.26 · β 11.00 · t 1.64*
#   US 09/1791~07/2011: Nov-Apr  2.24 · May-Oct 0.57 · β  1.67 · t 1.66*
LIT_B6 = {"kr": 11.00, "us": 1.67}
LIT_T = {"kr": 1.64, "us": 1.66}
LIT_MEAN6 = {"kr": (12.25, 1.26), "us": (2.24, 0.57)}   # (겨울, 여름) — 항등식 검산용
LIT_POOLED108 = 4.52       # 108개국 통합 (t 9.69) — **e* 로 쓰지 않는다**
# Table 2 표본 외(1998-09~2011-07) — **병기용**. e* 아님
LIT_OOS6 = {"kr": 12.82, "us": 4.90}
LIT_DIR = +1               # 문헌이 예측하는 부호 — 양수 유의면 채택

# ── 저장소 규약 (PROTOCOL §2 · 실행 위생이라 §0.3에서 유지) ────────────────
PRICE_FLOOR = {"kr": 1000.0, "us": 5.0}
DV_WIN = 20
DV_MIN = {"kr": None, "us": 2e6}       # PROTOCOL §2 (PIT) · KR 은 하한 없음
COST = {"kr": 0.0014, "us": 0.0005}    # **편도** (PROTOCOL §2 왕복의 절반)
MIN_STOCKS = 10                         # EW 일간수익을 내는 데 필요한 최소 종목
MIN_DAYS = 15                           # 월을 쓰려면 필요한 최소 유효 거래일
N_OUTLIER = 2                           # 견고성: 이상치 제외 개수 (M&P 2004 = 2개)

# **리밸일 우연의 CAGR 표준편차** — H-016 실측. 절대 성과 보고용
TIMING_LUCK_SD = {"kr": 4.02, "us": 1.43}

# ── 판정 칸과 임계 — **결과를 보기 전에 고정** (게이트 1차가 정리시킨 부분) ──
#
# **판정 칸은 ① 지수 하나다.** ② EW 유니버스는 **임계 없이 병기**한다.
#   ② 는 원문이 보고하지 않은 기초자산이라 §3.1-7(다) 한정 원리상 **보호할 원문
#   형태가 없고**, 따라서 (다)("검정력 50% 미만이면 실행하지 않는다")가 그대로
#   적용된다. ② 의 사전 검정력은 15.0% 다.
#   ② 를 판정 칸으로 유지하려면 "원문 미보고 기초자산도 §0 의 보호를 받는다"는
#   **예외를 자기 승인**해야 하는데, 그것은 §3.1-8(다)가 금지한 형태다.
JUDGE_CELLS = ("idx",)
#
# **임계는 가족 2칸 값(2.2414)을 그대로 쓴다.** 판정 칸이 1칸이면 1.96 으로
# 내려가지만 **그 완화를 취하지 않는다** — 칸을 줄인 이유가 (다)의 적용 범위이지
# 검출력이 아님을 **구조로** 못박기 위해서다. 1.96 을 썼다면 ① 의 검정력이
# 17.7% → 약 26% 로 올랐을 것이고, 그러면 "검출력을 위해 칸을 줄였다"를
# 반증할 수 없다. **더 엄격한 쪽으로만 벗어나므로 §3.1-8 위반이 아니다.**
FAMILY = 2
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414  ← 실제로 쓰는 임계
K_CRIT_UNUSED_FAMILY1 = _N.inv_cdf(1.0 - 0.05 / 2.0)   # 1.9600  ← **쓰지 않는다**


# ────────────────────────────────────────────────────────────── 계열

def is_winter(months: pd.PeriodIndex) -> np.ndarray:
    """`D_t` — 원문 407행. 달력 월만 보므로 **과거 봉만으로 판정된다**(§2-(2-ㄴ))."""
    return np.isin(np.asarray(months.month), WINTER)


def build_series(market: str) -> dict:
    """월별 **로그수익률** 두 계열을 만든다 — ① 지수 · ② EW 유니버스.

    ① 지수: 월말 종가 → 월말 종가 로그수익 (원문과 같은 형태).
    ② EW: 일별 동일가중 단순수익을 월 안에서 복리로 누적한 뒤 로그 (= 일간
       재조정 EW 포트폴리오의 월수익). 적격은 PROTOCOL §2 그대로.

    **마지막 역월은 부분 월이라 버린다.** 월 유효 거래일 `MIN_DAYS` 미만도 버린다.
    """
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(close.index))
    for k in piv:
        piv[k] = piv[k].reindex(index=close.index, columns=close.columns)

    cl = piv["close"].to_numpy(float)
    op = piv["open"].to_numpy(float)
    ok_bar = np.isfinite(cl) & (cl > 0) & np.isfinite(op) & (op > 0)   # PROTOCOL §1
    clm = np.where(ok_bar, cl, np.nan)

    dret = np.full_like(clm, np.nan)
    with np.errstate(all="ignore"):
        dret[1:] = clm[1:] / clm[:-1] - 1.0
    dret[~np.isfinite(dret)] = np.nan

    # 적격 — 가격 하한 + PIT 거래대금(전일 기준으로 시프트해 look-ahead 제거)
    dv = (piv["close"] * piv["volume"]).rolling(DV_WIN, min_periods=DV_WIN).mean()
    dv = dv.shift(1).to_numpy(float)
    elig = ok_bar & (clm >= PRICE_FLOOR[market])
    if DV_MIN[market] is not None:
        elig &= np.isfinite(dv) & (dv >= DV_MIN[market])
    elig[0] = False

    ew_d = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        r = dret[i][elig[i] & np.isfinite(dret[i])]
        if len(r) >= MIN_STOCKS:
            ew_d[i] = r.mean()

    b = loading.load_bench(market).set_index("date")["close"].reindex(close.index).ffill()
    bv = b.to_numpy(float)

    mk = pd.PeriodIndex(idx, freq="M")
    months = pd.PeriodIndex(sorted(set(mk)), freq="M")[:-1]     # 마지막 부분 월 제외

    # ① 지수 — 월말 종가 로그수익
    last = pd.Series(bv, index=mk).groupby(level=0).last()
    lv = np.log(last.reindex(months).to_numpy(float))
    prev = np.log(last.reindex(months - 1).to_numpy(float))
    with np.errstate(all="ignore"):
        r_idx = (lv - prev) * 100.0
    r_idx[~np.isfinite(r_idx)] = np.nan

    # ② EW — 월 안에서 일별 복리 → 로그
    r_ew = np.full(len(months), np.nan)
    n_days = np.zeros(len(months), dtype=int)
    for j, p in enumerate(months):
        sel = np.flatnonzero(mk == p)
        v = ew_d[sel]
        v = v[np.isfinite(v)]
        n_days[j] = len(v)
        if len(v) >= MIN_DAYS:
            r_ew[j] = np.log(np.prod(1.0 + v)) * 100.0

    return {"market": market, "months": months, "idx": r_idx, "ew": r_ew,
            "n_days": n_days, "dates": idx, "bench": bv, "mk": mk,
            "elig_n": elig.sum(axis=1), "ew_daily": ew_d}


def halloween_x(r: np.ndarray, months: pd.PeriodIndex) -> pd.Series:
    """원문 식 (1) 의 `β` 를 **평균으로 갖는 월별 계열**을 만든다.

        p = n_W / n
        x_t = r_t · D_t / p − r_t · (1 − D_t) / (1 − p)
        mean(x) = r̄_W − r̄_S      (**정확히 성립** — selftest 가 확인)

    더미회귀의 계수는 곧 두 집단 평균의 차이므로(원문 312행도
    `a regression with a dummy variable is nothing else than a difference in mean`),
    이 표현을 쓰면 PROTOCOL §3 의 **6종 클러스터/DK 추정량을 그대로 적용**할 수 있다.
    `p` 는 달력으로 결정되는 상수라 추정 오차를 싣지 않는다.

    반환 단위는 **%p/월**. 원문 β 척도로 보려면 × 6 한다.
    """
    ok = np.isfinite(r)
    rr, mm = r[ok], months[ok]
    if len(rr) == 0:
        return pd.Series(dtype=float)
    d = is_winter(mm).astype(float)
    p = d.mean()
    if p <= 0.0 or p >= 1.0:
        return pd.Series(dtype=float)
    x = rr * d / p - rr * (1.0 - d) / (1.0 - p)
    return pd.Series(x, index=pd.PeriodIndex(mm, freq="M"))


def season_blocks(r: np.ndarray, months: pd.PeriodIndex) -> pd.Series:
    """**견고성 — 원문 §5.3(반기 데이터).** 겨울/여름 6개월 블록의 누적 로그수익.

    반환은 블록 계열을 `halloween_x` 와 같은 방식으로 처리한 것 — 평균이
    `겨울 블록 평균 − 여름 블록 평균`(= 6개월 척도 β)이 된다.
    **블록이 6개월 전부 유효할 때만 쓴다**(부분 블록 금지).
    """
    vals, keys = [], []
    for j, p in enumerate(months):
        if p.month not in (11, 5):                 # 블록 시작 월만
            continue
        sel = [j + k for k in range(SEASON_LEN)]
        if sel[-1] >= len(months):
            continue
        if months[sel[-1]] != p + (SEASON_LEN - 1):
            continue                                # 달이 이어지지 않으면 버린다
        seg = r[sel]
        if not np.isfinite(seg).all():
            continue
        vals.append(seg.sum())                      # 로그수익이므로 합 = 누적
        keys.append(p)
    if not vals:
        return pd.Series(dtype=float)
    a = np.asarray(vals, dtype=float)
    ix = pd.PeriodIndex(keys, freq="M")
    d = (ix.month == 11).astype(float)
    p_ = d.mean()
    if p_ <= 0.0 or p_ >= 1.0:
        return pd.Series(dtype=float)
    return pd.Series(a * d / p_ - a * (1.0 - d) / (1.0 - p_), index=ix)


def drop_outliers(r: np.ndarray, n: int = N_OUTLIER) -> np.ndarray:
    """**견고성 — Maberly & Pierce (2004).** `|r|` 최대 `n` 개월을 결측 처리한다.

    M&P 는 1987-10 · 1998-08 **두 개**를 지목했고 그 둘을 더미로 흡수하면
    효과가 사라진다고 보고했다(mp.txt 468-470행). 우리 표본(2011~2026)에는
    그 두 달이 없으므로 **같은 개수(2개)를 크기 기준으로 대칭 제외**한다.
    **사전 지정 규칙이며 결과를 보고 고르지 않는다.**
    """
    out = r.copy()
    ok = np.flatnonzero(np.isfinite(out))
    if len(ok) <= n:
        return out
    worst = ok[np.argsort(-np.abs(out[ok]))[:n]]
    out[worst] = np.nan
    return out


def simple_returns(S: dict, key: str) -> np.ndarray:
    """**견고성 — 로그 대신 단순수익률.** 원문은 로그를 쓰므로 이쪽이 이탈 축이다."""
    return (np.exp(S[key] / 100.0) - 1.0) * 100.0


# ───────────────────────────────────────────────── 전략 형태 (임계 없이 병기)

def strategy_weights(S: dict) -> np.ndarray:
    """**11월 첫 거래일 종가 매수 → 5월 첫 거래일 종가 청산** 의 일별 비중.

    원문은 "10월 말 매수 / 4월 말 매도"인데 그것은 **기간의 끝 기준**이라
    t 시점에 판정할 수 없다(PROTOCOL §4-(a) 항목 3-(2-ㄴ)). **시작 기준으로
    재명세**했고, 경계 하루의 차이는 **근사**다(사전등록 §2.1).

    오늘이 11월의 첫 관측 거래일인지는 **과거 봉만으로** 안다 → look-ahead 없다.
    """
    mk = S["mk"]
    n = len(S["dates"])
    first = {}
    for i, p in enumerate(mk):
        first.setdefault(p, i)
    w = np.zeros(n)
    hold = False
    for i, p in enumerate(mk):
        if first.get(p) == i:                       # 그 달의 첫 거래일
            if p.month == 11:
                hold = True
            elif p.month == 5:
                hold = False
        # 진입일 **종가**에 사므로 그날 수익은 못 번다 → 다음 봉부터 보유
        w[i] = 1.0 if hold else 0.0
    out = np.zeros(n)
    out[1:] = w[:-1]
    return out


def strategy_stats(S: dict, key: str) -> dict:
    """전략의 **회전율·비용만** 낸다 — 수익은 `--run` 에서만 낸다."""
    w = strategy_weights(S)
    dw = float(np.abs(np.diff(np.concatenate([[0.0], w, [0.0]]))).sum())
    yrs = (S["dates"][-1] - S["dates"][0]).days / 365.25
    rt = dw / 2.0 / yrs                             # 왕복 횟수/년
    return {"sigma_dw": dw, "round_trips_per_year": rt,
            "cost_annual": COST[S["market"]] * dw / yrs * 100.0,
            "days_held": float(w.mean()), "years": yrs}


# ─────────────────────────────────────────────────────────────── 명령

def _pc():
    import pooled_clustering as PC
    return PC


_S: dict = {}


def _series() -> dict:
    if not _S:
        for mk in ("kr", "us"):
            _S[mk] = build_series(mk)
    return _S


CELLS = (("① 지수 (KR ^KS11 · US ^GSPC)", "idx"),
         ("② EW 유니버스", "ew"))


def pooled_e_star(n_kr: int, n_us: int) -> float:
    """표본가중 `e*` (%p/**월**). 원문 Table 3 **국가별** β 를 쓴다.

    **108개국 통합 4.52 를 쓰지 않는다** — 그것은 108개국 통합이고 우리는 2개
    시장이다(registry 우려 4). β 는 6개월 척도이므로 `/ SEASON_LEN` 으로 내린다.
    """
    return (n_kr * LIT_B6["kr"] + n_us * LIT_B6["us"]) / (n_kr + n_us) / SEASON_LEN


def cmd_power():
    PC = _pc()
    print("=" * 100)
    print("[H-032] 사전 검출력 — **SE·표본·회전율만**. 평균·t·부호는 출력 경로가 없다.")
    print(f"  **판정 칸 = {JUDGE_CELLS} 하나** · ② EW 는 임계 없이 병기(§3.1-7(다) 한정 원리)")
    print(f"  임계는 가족 {FAMILY}칸 값 |t| > {K_CRIT:.4f} 를 그대로 쓴다"
          f" — 1칸 완화값 {K_CRIT_UNUSED_FAMILY1:.4f} 는 **취하지 않는다**"
          f" · direction={LIT_DIR:+d}")
    print(f"  문헌 e* (원문 Table 3, **6개월 %p**): KR {LIT_B6['kr']} (t {LIT_T['kr']})"
          f" · US {LIT_B6['us']} (t {LIT_T['us']})")
    print(f"  ⚠️ 통합 108개국 {LIT_POOLED108} 는 **e* 로 쓰지 않는다** (108개국 통합이다)")
    print("  검정력은 PROTOCOL §3 개정판대로 **6종 중 SE 최대(= 최저 검정력)** 기준.")
    print("=" * 100)

    for mk, S in _series().items():
        w = loading.staleness_warning(loading.load_prices(mk))
        if w:
            print(f"\n[캐시] {mk.upper()}: {w}")
    for mk, S in _series().items():
        m = S["months"]
        nb = int(np.isfinite(S["idx"]).sum())
        ne = int(np.isfinite(S["ew"]).sum())
        w = is_winter(m)
        st = strategy_stats(S, "idx")
        print(f"\n── {mk.upper()} ── 월 {len(m)}개 ({m[0]}~{m[-1]})"
              f" · 겨울 {int(w.sum())} · 여름 {int((~w).sum())}")
        print(f"   유효 월: ① 지수 {nb} · ② EW {ne}"
              f" (EW 월 유효 거래일 중위 {int(np.median(S['n_days']))}"
              f" · 적격 종목 중위 {int(np.median(S['elig_n']))})")
        print(f"   반기 블록: ① {len(season_blocks(S['idx'], m))}"
              f" · ② {len(season_blocks(S['ew'], m))}")
        print(f"   전략 형태: 왕복 {st['round_trips_per_year']:.2f}회/년"
              f" · 비용 연 {st['cost_annual']:.4f}%p"
              f" · 보유 비중 {st['days_held']*100:.1f}%")

    for tag, key in CELLS:
        ser = {mk: halloween_x(S[key], S["months"]) for mk, S in _series().items()}
        ser = {k: v for k, v in ser.items() if len(v) > 12}
        if len(ser) < 2:
            print(f"\n[{tag}] 산출 불가")
            continue
        est = PC.all_estimates(ser)
        ses = {k: est[k] for k in PC.VOTERS}
        worst = max(ses, key=lambda k: ses[k])
        n_kr, n_us = len(ser["kr"]), len(ser["us"])
        e_star = pooled_e_star(n_kr, n_us)
        d = e_star / est[worst]
        pw = _N.cdf(d - K_CRIT) + _N.cdf(-d - K_CRIT)
        role = "**판정 칸**" if key in JUDGE_CELLS else "임계 없이 병기"
        print(f"\n[통합 {tag}] ({role}) n={est['n']}개월 · 겹치는 달 {est['overlap']}"
              f" · ρ={est['rho']:+.4f}")
        print(f"   G(달)={est['G_달']} G(분기)={est['G_분기']} G(연)={est['G_연']}")
        for k in PC.VOTERS:
            print(f"   {k:<10} SE {est[k]:.4f}%p/월  (6개월 척도 {est[k]*SEASON_LEN:.4f})")
        print(f"   naive      SE {est['naive']:.4f}%p/월 (판정에 투표하지 않음)")
        print(f"   표본가중 e* = ({n_kr}×{LIT_B6['kr']} + {n_us}×{LIT_B6['us']})"
              f"/{n_kr + n_us}/6 = **{e_star:.4f}%p/월**"
              f" (6개월 척도 {e_star*SEASON_LEN:.4f})")
        print(f"   **SE 최대 = {worst} ({est[worst]:.4f}) → 검정력 = {pw*100:.1f}%**")
        # 견고성 축의 SE 도 미리 낸다 (사전등록한 축은 사전등록 시점에 돌아가야 한다)
        for name, sr in (("반기 블록(원문 §5.3)",
                          {mk: season_blocks(S[key], S["months"])
                           for mk, S in _series().items()}),
                         (f"이상치 {N_OUTLIER}개 제외(M&P 2004)",
                          {mk: halloween_x(drop_outliers(S[key]), S["months"])
                           for mk, S in _series().items()}),
                         ("단순수익률",
                          {mk: halloween_x(simple_returns(S, key), S["months"])
                           for mk, S in _series().items()})):
            sr = {k: v for k, v in sr.items() if len(v) > 4}
            if len(sr) < 2:
                print(f"   [견고성] {name:<28} 산출 불가")
                continue
            e2 = PC.all_estimates(sr)
            w2 = max(PC.VOTERS, key=lambda k: e2[k])
            print(f"   [견고성] {name:<28} n={e2['n']:>3}"
                  f" · SE 최대 {e2[w2]:.4f} ({w2})"
                  f" · G(연)={e2['G_연']}")


def cmd_selftest():
    ok = []

    def chk(name, cond):
        ok.append((f"{len(ok) + 1:02d}", name, bool(cond)))

    # ── 원문 사양 고정값 ──────────────────────────────────────────────
    chk("겨울 = 11·12·1·2·3·4월 (원문 `November through April`)",
        WINTER == (11, 12, 1, 2, 3, 4))
    chk("가족 2칸 → K_CRIT ≈ 2.2414", abs(K_CRIT - 2.2414) < 5e-4)
    chk("문헌 방향 +1 (양수 유의면 채택)", LIT_DIR == +1)
    chk("e* 는 Table 3 국가별 값 (KR 11.00 · US 1.67)",
        LIT_B6 == {"kr": 11.00, "us": 1.67})
    chk("108개국 통합 4.52 는 별도 상수로만 두고 e* 에 쓰지 않는다",
        LIT_POOLED108 == 4.52 and LIT_POOLED108 not in LIT_B6.values())
    # **e* 산식을 실호출로 검사한다** — 상수 자리만 보는 것으로는 부족하다(게이트 1차 지적).
    #   (146×11.00 + 178×1.67)/324/6 = 1903.26/1944 = 0.979042…
    chk("e* 산식 실호출: pooled_e_star(146,178) = 0.9790%p/월 (손계산 대조)",
        abs(pooled_e_star(146, 178) - (146 * 11.00 + 178 * 1.67) / 324 / 6) < 1e-12
        and abs(pooled_e_star(146, 178) - 0.9790432) < 1e-6)
    chk("e* 는 6개월 척도로 되돌리면 5.8742593",
        abs(pooled_e_star(146, 178) * SEASON_LEN - 5.8742593) < 1e-6)
    chk("e* 산식이 108개국 통합값을 참조하지 않는다 (소스 검사)",
        "LIT_POOLED108" not in __import__("inspect").getsource(pooled_e_star))
    # 한쪽 시장만 있으면 그 시장의 값이 그대로 나온다 (가중 평균의 극단값)
    chk("e*: US 표본이 0이면 KR 값 11.00/6 = 1.8333",
        abs(pooled_e_star(100, 0) - 11.00 / 6) < 1e-12)
    chk("e*: KR 표본이 0이면 US 값 1.67/6 = 0.2783",
        abs(pooled_e_star(0, 100) - 1.67 / 6) < 1e-12)

    # ── 판정 칸과 임계 (게이트 1차가 정리시킨 부분) ──────────────────────
    chk("판정 칸은 ① 지수 하나 — ② EW 는 임계 없이 병기", JUDGE_CELLS == ("idx",))
    chk("임계는 가족 2칸 값 2.2414 를 그대로 쓴다 (1칸 완화 1.96 을 취하지 않는다)",
        abs(K_CRIT - 2.2414) < 5e-4
        and abs(K_CRIT_UNUSED_FAMILY1 - 1.9600) < 5e-4
        and K_CRIT > K_CRIT_UNUSED_FAMILY1)
    # **1칸 완화값이 실제로 판정·검정력에 안 쓰이는지 소스로 검사한다** — 상수 값과
    # 대소만 보면 나중에 누가 그것을 검정력 계산에 넣어도 통과한다(게이트 2차 지적).
    _psrc = __import__("inspect").getsource(cmd_power)
    chk("검정력은 K_CRIT 로 계산하고 1칸 완화값은 안내 출력 1회에만 등장 (소스 검사)",
        _psrc.count("K_CRIT_UNUSED_FAMILY1") == 1
        and "_N.cdf(d - K_CRIT)" in _psrc)

    # ── 원문 Table 3 척도 항등식 (β = 겨울 − 여름, 둘 다 월수익×6) ───────
    for mk in ("kr", "us"):
        wsum, ssum = LIT_MEAN6[mk]
        chk(f"Table 3 {mk.upper()}: {wsum} − {ssum} = β {LIT_B6[mk]} (허용 0.01)",
            abs((wsum - ssum) - LIT_B6[mk]) <= 0.01)
    chk("β 는 6개월 척도 — 월 척도로 내리려면 /6 (KR 11.00 → 1.8333)",
        abs(LIT_B6["kr"] / SEASON_LEN - 1.83333) < 1e-4)

    # ── is_winter ────────────────────────────────────────────────────
    mm = pd.PeriodIndex(pd.date_range("2020-01-01", periods=12, freq="MS"), freq="M")
    got = is_winter(mm)
    want = np.array([m in WINTER for m in range(1, 13)])
    chk("is_winter 12개월 전수 일치 (1~4월·11~12월만 True)", (got == want).all())
    chk("겨울 6개월 · 여름 6개월", got.sum() == 6 and (~got).sum() == 6)

    # ── halloween_x: 평균이 정확히 겨울−여름 평균 차인가 ─────────────────
    rng = np.random.default_rng(20260819)
    for trial in range(200):
        n = int(rng.integers(14, 60))
        m2 = pd.PeriodIndex(pd.date_range("2015-03-01", periods=n, freq="MS"), freq="M")
        r2 = rng.normal(0.5, 5.0, n)
        d2 = is_winter(m2)
        if d2.all() or (~d2).any() is False:
            continue
        want2 = r2[d2].mean() - r2[~d2].mean()
        got2 = halloween_x(r2, m2).mean()
        if abs(got2 - want2) > 1e-10:
            chk(f"halloween_x 평균 = 겨울−여름 (trial {trial})", False)
            break
    else:
        chk("halloween_x 평균 = 겨울−여름 평균 차 (난수 200회, 오차 1e-10)", True)

    # 손계산 소형 예제 — 2020-01(겨울) 10 · 2020-05(여름) 2 · 2020-06(여름) 4
    m3 = pd.PeriodIndex(["2020-01", "2020-05", "2020-06"], freq="M")
    r3 = np.array([10.0, 2.0, 4.0])
    #   p = 1/3 → x = [10/(1/3), −2/(2/3), −4/(2/3)] = [30, −3, −6] → 평균 7
    #   겨울 10 − 여름 3 = 7 ✓
    x3 = halloween_x(r3, m3)
    chk("소형 예제: x = (30, −3, −6) · 평균 7 = 10 − 3",
        np.allclose(x3.to_numpy(), [30.0, -3.0, -6.0]) and abs(x3.mean() - 7.0) < 1e-12)

    # OLS 더미회귀 계수와 정확히 같은가 (원문 312행: 더미회귀 = 평균 차)
    n4 = 48
    m4 = pd.PeriodIndex(pd.date_range("2016-01-01", periods=n4, freq="MS"), freq="M")
    r4 = rng.normal(0.3, 4.0, n4)
    X4 = np.column_stack([np.ones(n4), is_winter(m4).astype(float)])
    b4, *_ = np.linalg.lstsq(X4, r4, rcond=None)
    chk("halloween_x 평균 = 더미회귀 OLS 계수 β (원문 식 1)",
        abs(halloween_x(r4, m4).mean() - b4[1]) < 1e-10)

    # 결측이 섞여도 성립하는가 (유효 월만 쓰고 p 를 다시 잰다)
    r5 = r4.copy()
    r5[[3, 7, 20, 31]] = np.nan
    okm = np.isfinite(r5)
    want5 = r5[okm & is_winter(m4)].mean() - r5[okm & ~is_winter(m4)].mean()
    chk("결측 월이 있어도 평균 = 겨울−여름 (유효 월만)",
        abs(halloween_x(r5, m4).mean() - want5) < 1e-10)

    # 극단값: 겨울·여름 수익이 같으면 0
    r6 = np.where(is_winter(m4), 3.0, 3.0)
    chk("겨울=여름이면 x 의 평균 = 0", abs(halloween_x(r6, m4).mean()) < 1e-12)
    # 겨울만 +5, 여름 0 이면 평균 = 5
    r7 = np.where(is_winter(m4), 5.0, 0.0)
    chk("겨울 +5 · 여름 0 이면 평균 = 5", abs(halloween_x(r7, m4).mean() - 5.0) < 1e-12)
    # 전부 겨울이면 산출 불가(빈 계열)
    m8 = pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    chk("전부 겨울이면 빈 계열 (p=1 → 산출 불가)",
        len(halloween_x(np.array([1.0, 2.0, 3.0]), m8)) == 0)

    # ── season_blocks ────────────────────────────────────────────────
    m9 = pd.PeriodIndex(pd.date_range("2019-11-01", periods=24, freq="MS"), freq="M")
    r9 = np.zeros(24)
    r9[is_winter(m9)] = 1.0                      # 겨울 월마다 +1 → 블록 합 6
    sb = season_blocks(r9, m9)
    chk("반기 블록: 24개월 → 블록 4개 (겨울2·여름2)", len(sb) == 4)
    chk("반기 블록 평균 = 겨울블록 6 − 여름블록 0 = 6", abs(sb.mean() - 6.0) < 1e-12)
    r10 = r9.copy(); r10[3] = np.nan             # 첫 겨울 블록 안에 결측
    chk("블록 안에 결측 1개면 그 블록을 버린다 (부분 블록 금지)",
        len(season_blocks(r10, m9)) == 3)
    # **`season_blocks` 를 실제로 호출해** 합산(누적)인지 본다 — 종전 검사는 자기
    # 입력 배열에 numpy `.sum()` 을 건 것이라 구현이 바뀌어도 통과했다(게이트 1차 지적).
    #   겨울 블록 2개(각 6.0) · 여름 블록 2개(각 0.0) → 평균 6.0
    #   첫 겨울 달을 1.0 → 2.0 으로 올리면 그 블록이 7.0 → 겨울 평균 6.5 → 전체 6.5
    r9b = r9.copy()
    r9b[np.flatnonzero(is_winter(m9))[0]] = 2.0
    chk("반기 블록은 로그수익의 **합**(누적) — 겨울 한 달 +1 → 그 블록 6→7 → 평균 6.5",
        abs(float(season_blocks(r9b, m9).mean()) - 6.5) < 1e-12)

    # ── drop_outliers ────────────────────────────────────────────────
    r11 = np.array([1.0, -30.0, 2.0, 25.0, 3.0, -4.0])
    d11 = drop_outliers(r11)
    chk("이상치 2개 제외 = |r| 최대 두 개(−30, 25)만 결측",
        np.isnan(d11[1]) and np.isnan(d11[3])
        and np.isfinite(d11[[0, 2, 4, 5]]).all())
    chk("이상치 제외 개수는 M&P 2004 의 2개", N_OUTLIER == 2)
    chk("표본이 제외 개수 이하면 그대로 둔다",
        np.isfinite(drop_outliers(np.array([1.0, 2.0]))).all())

    # ── 단순수익률 변환 (로그 ↔ 단순) ─────────────────────────────────
    s12 = simple_returns({"idx": np.array([0.0, 10.0, -10.0])}, "idx")
    chk("로그 0% → 단순 0%", abs(s12[0]) < 1e-12)
    chk("로그 +10% → 단순 +10.517%", abs(s12[1] - (np.exp(0.1) - 1) * 100) < 1e-12)
    # 라벨이 assert 와 반대였다(게이트 1차 지적). **음수 쪽에서 단순 > 로그**이고,
    # 값 자체(−9.516)도 검사한다.
    chk("로그 −10% → 단순 −9.516% (음수 쪽에서 **단순 > 로그**)",
        abs(s12[2] - (np.exp(-0.1) - 1) * 100) < 1e-12 and s12[2] > -10.0)

    # ── 전략 비중: (2-ㄴ) 시작 기준 · 진입일 종가라 다음 봉부터 보유 ───────
    dts = pd.DatetimeIndex(pd.bdate_range("2020-10-28", "2021-05-05"))
    S13 = {"dates": dts, "mk": pd.PeriodIndex(dts, freq="M"), "market": "us"}
    w13 = strategy_weights(S13)
    i_nov = int(np.flatnonzero(pd.PeriodIndex(dts, freq="M") == pd.Period("2020-11"))[0])
    i_may = int(np.flatnonzero(pd.PeriodIndex(dts, freq="M") == pd.Period("2021-05"))[0])
    chk("11월 첫 거래일에는 아직 미보유 (그날 종가에 산다)", w13[i_nov] == 0.0)
    chk("11월 둘째 거래일부터 보유", w13[i_nov + 1] == 1.0)
    chk("5월 첫 거래일까지 보유하고 그 다음 봉부터 미보유",
        w13[i_may] == 1.0 and w13[i_may + 1] == 0.0)
    chk("10월 구간은 전부 미보유", w13[:i_nov].sum() == 0.0)
    st13 = strategy_stats(S13, "idx")
    chk("Σ|Δw| = 2.0 (연 1왕복)", abs(st13["sigma_dw"] - 2.0) < 1e-12)

    # ── 비용 규약 (PROTOCOL §2 왕복의 절반이 편도) ──────────────────────
    chk("COST 는 PROTOCOL §2 왕복(KR 0.28% · US 0.10%)의 절반(편도)",
        abs(COST["kr"] - 0.0028 / 2) < 1e-12 and abs(COST["us"] - 0.0010 / 2) < 1e-12)

    # ── 리밸일 잡음 상수 (H-016) ───────────────────────────────────────
    chk("리밸일 CAGR 표준편차 = H-016 실측 (KR 4.02 · US 1.43)",
        TIMING_LUCK_SD == {"kr": 4.02, "us": 1.43})

    # ── --power 가 점추정치를 내지 않는가 (사전등록 오염 방지) ────────────
    import inspect
    src = inspect.getsource(cmd_power)
    chk("--power 에 verdict 호출 없음", "verdict" not in src)
    chk("--power 에 est['mu'] · mu 인덱싱 없음",
        "'mu'" not in src and '"mu"' not in src)
    # 라벨 주의: `--power` 는 SE·개수 외에 **회전율·비용·보유비중**도 출력한다.
    # 여기서 막는 것은 **수익률 계열의 평균**(= 점추정치)이다 (게이트 2차 지적).
    chk("--power 에 .mean() 없음 — 수익률 계열의 평균을 출력할 경로가 없다",
        ".mean()" not in src)
    # 문자열을 쪼개 쓰는 이유: 통째로 적으면 **이 검사문 자신이** 모듈 소스에
    # 걸려 항상 실패한다(자기참조). 실제 정의만 잡히도록 한다.
    chk("--run 은 아직 없다 (게이트 통과 후 작성)",
        ("def cmd_" + "run") not in inspect.getsource(sys.modules[__name__]))

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
    else:
        print(__doc__)
        sys.exit(2)
