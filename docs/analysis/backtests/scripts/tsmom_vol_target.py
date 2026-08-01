"""H-012 — 변동성 타깃 시계열 모멘텀 (랭킹 없는 월간 리밸런싱 포트폴리오).

설계 문서: docs/analysis/backtests/2026-08-01-tsmom-vol-target.md
규칙·대조군·사전 판정 기준은 전부 그 문서 §2~§3을 따른다.

**btlib를 쓰지 않고 새로 짠 부분과 그 이유**
btlib.engine은 단일 거래 평가기라 포트폴리오를 표현할 수 없고, H-009(주간·ATR)·
H-010(월간·동일비중)의 시뮬레이터는 이 가설의 매매 절차(가변 종목수·역변동성 가중·
동적 익스포저·잔여 현금 재배분)와 달라 재사용할 수 없다. 데이터 로딩(`btlib.loading`)·
레짐(`btlib.regime`)·비용 원천값(`btlib.costs`)은 btlib를 그대로 쓴다.

**익스포저 계산 구조 (문서 §2.1·§2.5)**
E는 "무스케일 계열"(= 그 그룹의 E=1 쌍둥이)의 직전 126 **투자일** 실현변동성으로 정한다.
따라서 E=1 계열을 먼저 돌려 일간수익률을 얻고, 그것으로 E 시계열을 만든 뒤 본 계열을
돌린다. A의 무스케일 계열은 곧 그룹 D이므로 D를 그대로 재사용한다.

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/tsmom_vol_target.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/tsmom_vol_target.py kr
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/tsmom_vol_target.py us
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # docs/analysis
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # 저장소 루트

from btlib import costs, loading, regime  # noqa: E402

# ── 문서 §2 고정 파라미터 ────────────────────────────────────────────────
LB_MOM = 252            # 절대 모멘텀 창
SIG_WIN, SIG_MIN = 60, 55   # σ60: 종가 61개 사이 수익률 60개, 유효 55 이상
VOL_WIN = 126           # 포트폴리오 실현변동성: 투자일 126개
VOL_TARGET = 0.12       # 연 12%
VOL_TARGET_ALT = 0.08   # 민감도(판정 미사용)
MIN_BARS = LB_MOM + 1   # 253
DV_WIN = 60
SMA_BENCH = 200
START_EQUITY = 1e8
MIN_E_LT1_MONTHS = 12   # ③ 판정 최소 표본 (문서 §3)

MARKET_CFG = {
    "us": {"universe_n": 500, "cost_pct": costs.COST_PCT["US"] / 2},   # 편도
    "kr": {"universe_n": 300, "cost_pct": costs.COST_PCT["KR"] / 2},
}
#: 그룹 정의 — (역변동성 가중?, 변동성 타깃?, 절대 모멘텀 필터?)
GROUP_SPEC = {
    "A": dict(invvol=True,  target=True,  mom=True),
    "C": dict(invvol=False, target=True,  mom=True),    # 동일비중
    "D": dict(invvol=True,  target=False, mom=True),    # E=1
    "E": dict(invvol=True,  target=True,  mom=False),   # 전 종목
}


# ── 패널 ─────────────────────────────────────────────────────────────────

def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # 종가 '또는' 시가가 0·음수면 해당 봉만 NaN (문서 §2.1 — OR 조건).
    # KR 거래정지 봉은 open=0인데 close는 양수 이월이라 AND로 검사하면 못 잡는다.
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    # 일간수익률 — 공통 축 유지, 인접 두 봉 (NaN 드롭 후 pct_change 금지)
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0

    mom = np.full_like(c, np.nan)
    mom[LB_MOM:] = c[LB_MOM:] / c[:-LB_MOM] - 1.0

    rdf = pd.DataFrame(ret)
    sig = rdf.rolling(SIG_WIN, min_periods=SIG_MIN).std(ddof=1).to_numpy()

    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close": c, "close_ff": close.ffill().to_numpy(float),
            "open": piv["open"].to_numpy(float), "ret": ret,
            "mom": mom, "sigma": sig,
            "dollar_vol": (close * piv["volume"]).to_numpy(float),
            "bars": close.notna().cumsum().to_numpy(float)}


def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).max() if v >= start_i]


def invested_day_mask(n_days: int, rebs: list[int], bull: np.ndarray) -> np.ndarray:
    """투자일 마스크 — 리밸→리밸 블록 기준 (문서 §2.1).

    블록 = (리밸일, 다음 리밸일]. 그 블록의 지배 플래그는 **블록 시작 리밸일**의 레짐.
    달력 월로 라벨링하면 월말 리밸 스케줄과 한 달 어긋난다.
    """
    m = np.zeros(n_days, dtype=bool)
    for a, b in zip(rebs, rebs[1:] + [n_days - 1]):
        if bull[a]:
            m[a + 1:b + 1] = True
    return m


def exposure_series(daily_ret: np.ndarray, inv_mask: np.ndarray, rebs: list[int],
                    target: float) -> dict:
    """리밸일 i → E. 직전 126 투자일(판단일 t 제외)의 실현변동성 기준."""
    out = {}
    for i in rebs:
        hist = np.flatnonzero(inv_mask[:i] & np.isfinite(daily_ret[:i]))
        if len(hist) < VOL_WIN:
            out[i] = 1.0                      # 워밍업
            continue
        v = daily_ret[hist[-VOL_WIN:]].std(ddof=1) * np.sqrt(252)
        out[i] = 1.0 if not np.isfinite(v) or v <= 0 else float(min(1.0, target / v))
    return out


# ── 시뮬레이터 (신규 작성) ───────────────────────────────────────────────

def select(P, i, spec, cfg):
    """적격 종목과 원시 비중 w (합 1). 반환: (idx 배열, w 배열).

    **순서가 중요하다(문서 §2.2)**: ① 유동성 유니버스 = 거래대금 상위 N을 먼저 자르고
    ② 그 안에서 모멘텀·σ60 조건을 적용해 "적격 종목"을 얻는다. 순서를 뒤집으면
    (모멘텀 먼저 → 유동성 상위 N) 유니버스가 항상 상한 N으로 차버려 **보유 종목수가
    시장 상태에 따라 변한다는 가설의 핵심이 사라진다.**
    """
    px = P["close"][i]
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dvm = np.nanmedian(dv, axis=0)
    # ① 유동성 유니버스
    liq = np.isfinite(px) & (px > 0) & (P["bars"][i] >= MIN_BARS) & np.isfinite(dvm)
    if not liq.any():
        return np.array([], dtype=int), np.array([])
    keep = min(int(cfg["universe_n"]), int(liq.sum()))
    thr = np.sort(dvm[liq])[::-1][keep - 1]
    univ = liq & (dvm >= thr)
    # ② 적격 종목 — 유니버스 '중' 조건 충족분 전부 (개수 제한 없음)
    elig = univ & np.isfinite(P["sigma"][i]) & (P["sigma"][i] > 0)
    if spec["mom"]:
        elig &= np.isfinite(P["mom"][i]) & (P["mom"][i] > 0)
    sel = np.flatnonzero(elig)
    if len(sel) == 0:
        return sel, np.array([])
    raw = (1.0 / P["sigma"][i][sel]) if spec["invvol"] else np.ones(len(sel))
    return sel, raw / raw.sum()


def simulate(P, bull, group, cfg, cost_pct, exposures=None, exec_next_open=False,
             collect_ret=False) -> dict:
    """exposures: {리밸 인덱스: E}. None이면 E=1 고정(무스케일 쌍둥이)."""
    spec = GROUP_SPEC[group]
    close, dates, close_ff = P["close"], P["dates"], P["close_ff"]
    n_days, n_tk = close.shape
    tick = P["tickers"]

    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_MOM, 1)
    rebs = month_end_indices(dates, start_i)
    reb_set = set(rebs)

    shares = np.zeros(n_tk)
    cash = START_EQUITY
    equity = np.full(n_days, np.nan)
    invested = np.full(n_days, np.nan)
    n_pos = np.full(n_days, np.nan)
    traded, eq_sum, zombie = 0.0, 0.0, 0
    e_log, real_exp, w_dev = {}, [], []

    for i in range(rebs[0], n_days):
        val = float((shares * np.nan_to_num(close_ff[i])).sum())
        equity[i] = cash + val
        eq_sum += equity[i]
        invested[i] = val / equity[i] if equity[i] > 0 else np.nan
        n_pos[i] = int((shares > 0).sum())
        if i not in reb_set:
            continue

        exec_px = P["open"][i + 1] if (exec_next_open and i + 1 < n_days) else close[i]
        tradable = np.isfinite(exec_px) & (exec_px > 0)
        zombie += int(((shares > 0) & ~tradable).sum())

        E = 1.0 if (exposures is None or not spec["target"]) else exposures.get(i, 1.0)
        # (E, 실제 투자 여부) — ③ 판정 표본에서 '약세라 전액 현금인 달'을 빼기 위함
        e_log[i] = (E, bool(bull[i]))
        base_eq = equity[i]

        sel, w = select(P, i, spec, cfg) if bull[i] else (np.array([], dtype=int),
                                                          np.array([]))
        if not bull[i] or len(sel) == 0:       # ① 레짐 약세 또는 적격 0 → 전량 청산
            for j in np.flatnonzero(shares > 0):
                if tradable[j]:
                    cash += shares[j] * exec_px[j] * (1 - cost_pct / 100.0)
                    traded += shares[j] * exec_px[j]
                    shares[j] = 0.0
            continue
        order = sorted(range(len(sel)), key=lambda k: (-w[k], tick[sel[k]]))
        sel, w = sel[order], w[order]
        q = np.zeros(n_tk)
        tgt_amt = np.zeros(n_tk)
        for k, j in enumerate(sel):
            if not tradable[j]:
                continue
            tgt_amt[j] = base_eq * w[k] * E
            q[j] = np.floor(tgt_amt[j] / exec_px[j])

        # ③ 매도 — (a) 목표 밖 전량, (b) 초과분 부분
        in_target = np.zeros(n_tk, dtype=bool)
        in_target[sel] = True
        for j in np.flatnonzero(shares > 0):
            if not tradable[j]:
                continue                        # 좀비: 팔 수 없으니 보유 유지
            excess = shares[j] - (q[j] if in_target[j] else 0.0)
            if excess > 0:
                cash += excess * exec_px[j] * (1 - cost_pct / 100.0)
                traded += excess * exec_px[j]
                shares[j] -= excess

        # ④ 매수 — 집행 순서 키대로 q까지
        for j in sel:
            if not tradable[j] or shares[j] >= q[j]:
                continue
            need_q = q[j] - shares[j]
            cost_amt = need_q * exec_px[j] * (1 + cost_pct / 100.0)
            if cost_amt > cash:
                continue                        # 현금 부족 → 건너뜀
            cash -= cost_amt
            traded += need_q * exec_px[j]
            shares[j] += need_q

        # ⑤ 잔여 현금 재배분 — 정수 내림 누수만, 종목당 최대 1주, 1라운드
        held_target = float(sum(shares[j] * exec_px[j] for j in sel if tradable[j]))
        budget = base_eq * E - held_target
        if budget > 0:
            for j in sel:
                if not tradable[j]:
                    continue
                amt = exec_px[j] * (1 + cost_pct / 100.0)
                if amt > budget or amt > cash:
                    continue
                cash -= amt
                budget -= amt
                traded += exec_px[j]
                shares[j] += 1.0

        # 부가 기록 — 문서 §2.3-5가 요구한 것은 "**목표** 익스포저 대비 실현 익스포저"다
        realized = float(sum(shares[j] * exec_px[j] for j in sel if tradable[j]))
        real_exp.append(realized / (base_eq * E) if base_eq > 0 and E > 0 else np.nan)
        if E > 0:
            dev = [abs(shares[j] * exec_px[j] / base_eq - w[k] * E)
                   for k, j in enumerate(sel) if tradable[j]]
            if dev:
                w_dev.append(float(np.mean(dev)))

    eq_s = pd.Series(equity, index=pd.to_datetime(dates)).dropna()
    if len(eq_s) != n_days - rebs[0]:
        raise RuntimeError(f"equity 절단: {len(eq_s)} != {n_days - rebs[0]}")
    n_obs = max(len(eq_s), 1)
    years = max((eq_s.index[-1] - eq_s.index[0]).days / 365.25, 1e-9)
    out = {"equity": eq_s, "turnover": (traded / 2.0) / (eq_sum / n_obs) / years,
           "invested": float(np.nanmean(invested)), "n_pos": float(np.nanmean(n_pos)),
           "zombie": zombie, "E": e_log, "rebs": rebs,
           "real_exp": float(np.mean(real_exp)) if real_exp else np.nan,
           "w_dev": float(np.mean(w_dev)) if w_dev else np.nan}
    if collect_ret:
        r = np.full(n_days, np.nan)
        eqv = equity
        r[1:] = np.where(np.isfinite(eqv[:-1]) & (eqv[:-1] > 0),
                         eqv[1:] / eqv[:-1] - 1.0, np.nan)
        out["daily_ret"] = r
    return out


# ── 통계 ─────────────────────────────────────────────────────────────────

def monthly(s: pd.Series) -> pd.Series:
    return s.resample("ME").last().pct_change().dropna()


def stats(eq: pd.Series, bench_eq: pd.Series) -> dict:
    m, bm = monthly(eq), monthly(bench_eq)
    j = pd.concat([m, bm], axis=1, join="inner").dropna()
    exc = (j.iloc[:, 0] - j.iloc[:, 1]) * 100
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"n_months": len(exc), "exc_mean": exc.mean(), "exc_med": exc.median(),
            "cagr": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
            "vol": m.std() * np.sqrt(12) * 100,
            "sharpe": (m.mean() / m.std() * np.sqrt(12)) if m.std() > 0 else np.nan,
            "mdd": (eq / eq.cummax() - 1).min() * 100}


def delta_sharpe_ci(ra: pd.Series, rb: pd.Series) -> dict:
    """Memmel 보정 Jobson–Korkie — 월 단위 ΔSharpe와 95% CI (문서 §3).

    Var(ΔSR) = (1/T)[2 − 2ρ + ½SR_A² + ½SR_B² − SR_A·SR_B·ρ²]
    """
    j = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(j) < 3:
        return {"n": len(j), "d": np.nan, "lo": np.nan, "hi": np.nan}
    a, b = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
    T = len(a)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if sa <= 0 or sb <= 0:
        return {"n": T, "d": np.nan, "lo": np.nan, "hi": np.nan}
    SRa, SRb = a.mean() / sa, b.mean() / sb
    rho = float(np.corrcoef(a, b)[0, 1])
    var = (2 - 2 * rho + 0.5 * SRa ** 2 + 0.5 * SRb ** 2 - SRa * SRb * rho ** 2) / T
    se = np.sqrt(max(var, 0.0))
    d = SRa - SRb
    return {"n": T, "d": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se,
            "sharpe_a": SRa * np.sqrt(12), "sharpe_b": SRb * np.sqrt(12)}


def run_market(market: str, target: float = VOL_TARGET, quiet: bool = False):
    cfg = MARKET_CFG[market]
    P = build_panel(market)
    bench = loading.load_bench(market)
    end = min(P["dates"][-1], bench["date"].max())      # 문서 §4
    keep = P["dates"] <= end
    for k in ("close", "close_ff", "open", "ret", "mom", "sigma", "dollar_vol", "bars"):
        P[k] = P[k][keep]
    P["dates"] = P["dates"][keep]

    bull_map = regime.bull_map(bench, sma=SMA_BENCH)
    bull = np.array([bull_map.get(d, False) for d in P["dates"]])
    bser = bench.set_index(pd.to_datetime(bench["date"]))["close"].astype(float)

    # 1단계 — 무스케일(E=1) 쌍둥이로 각 그룹의 E 시계열 산출
    shadows, exposures = {}, {}
    for g in ("A", "C", "E"):
        sg = "D" if g == "A" else g          # A의 쌍둥이는 곧 그룹 D
        sh = simulate(P, bull, sg, cfg, cfg["cost_pct"], exposures=None,
                      collect_ret=True)
        shadows[g] = sh
        inv = invested_day_mask(len(P["dates"]), sh["rebs"], bull)
        exposures[g] = exposure_series(sh["daily_ret"], inv, sh["rebs"], target)

    # 2단계 — 본 계열
    out = {}
    for net in (True, False):
        cp = cfg["cost_pct"] if net else 0.0
        out[("D", net)] = simulate(P, bull, "D", cfg, cp, exposures=None)
        for g in ("A", "C", "E"):
            out[(g, net)] = simulate(P, bull, g, cfg, cp, exposures=exposures[g])

    a_eq = out[("A", True)]["equity"]
    bench_eq = bser.reindex(a_eq.index).ffill().dropna()
    a_eq = a_eq.reindex(bench_eq.index).dropna()
    bench_eq = bench_eq.reindex(a_eq.index)

    rows = {}
    for g in ("A", "C", "D", "E"):
        eq = out[(g, True)]["equity"].reindex(a_eq.index).ffill()
        rows[g] = stats(eq, bench_eq)
        for k in ("turnover", "invested", "n_pos", "zombie", "real_exp", "w_dev"):
            rows[g][k] = out[(g, True)][k]
    rows["B"] = stats(bench_eq, bench_eq)
    rows["B"].update(turnover=0.0, invested=1.0, n_pos=np.nan, zombie=0,
                     real_exp=1.0, w_dev=0.0)

    if quiet:
        return rows, out, exposures, P, bull, a_eq, bench_eq

    print(f"\n{'='*92}\n[{market.upper()}] 유니버스 {len(P['tickers']):,}종목 · "
          f"기준일 {P['dates'][-1]} · 유동성 상위 {cfg['universe_n']} · "
          f"타깃 {target*100:.0f}% · 편도비용 {cfg['cost_pct']}%\n{'='*92}")
    print(f"{'그룹':<12} {'월수':>4} {'월초과평균':>9} {'월초과중위':>9} {'CAGR':>7} "
          f"{'변동성':>6} {'Sharpe':>6} {'MDD':>7} {'회전율':>7} {'투자비중':>7} {'종목수':>6}")
    print("-" * 100)
    label = {"A": "A 전체", "B": "B 벤치마크", "C": "C 동일비중",
             "D": "D 타깃없음", "E": "E 필터없음"}
    for g in ("A", "B", "C", "D", "E"):
        s = rows[g]
        print(f"{label[g]:<12} {s['n_months']:>4} {s['exc_mean']:>8.3f}% {s['exc_med']:>8.3f}% "
              f"{s['cagr']:>6.2f}% {s['vol']:>5.1f}% {s['sharpe']:>6.2f} "
              f"{s['mdd']:>6.1f}% {s['turnover']:>6.2f}x {s['invested']*100:>6.1f}% "
              f"{s['n_pos']:>5.0f}")

    def mret(g, net=True):
        return monthly(out[(g, net)]["equity"].reindex(a_eq.index).ffill())

    # E<1 달 라벨 — 월 M의 수익률을 지배하는 E는 M−1 말 리밸일의 E.
    # 리밸일이 달력 말일이 아닐 수 있으므로 Period로 "다음 달"을 잡는다
    # (MonthEnd(1)은 말일이 아닌 날짜를 같은 달로 매핑해 중복 라벨을 만든다).
    def e_lt1_months(g):
        """E<1 **이면서 실제로 투자한** 달. 약세라 전액 현금인 달은 A≡D라 차분이
        정확히 0이 되어 ρ를 1로 밀고 CI를 좁힌다(§3이 E<1 필터를 둔 바로 그 이유)."""
        idx, lab = [], []
        for i, (E, inv) in sorted(out[(g, True)]["E"].items()):
            nxt = (pd.Period(pd.Timestamp(P["dates"][i]), freq="M") + 1)
            idx.append(nxt.end_time.normalize())
            lab.append(bool(inv) and E < 1 - 1e-12)
        s = pd.Series(lab, index=pd.DatetimeIndex(idx))
        return s[~s.index.duplicated(keep="last")]

    print(f"\n[핵심 차분]")
    ac = delta_sharpe_ci(mret("A"), mret("C"))
    print(f"  ② ΔSharpe(A−C) 비용후: {ac['d']:+.4f} (월) · "
          f"95% CI [{ac['lo']:+.4f}, {ac['hi']:+.4f}] · n={ac['n']} "
          f"→ {'✓' if ac['d'] > 0 else '✗'}  {'(CI가 0 포함)' if ac['lo'] <= 0 <= ac['hi'] else '(CI 0 제외)'}")
    ad_all = delta_sharpe_ci(mret("A", False), mret("D", False))
    m_lt1 = e_lt1_months("A").reindex(mret("A", False).index).fillna(False)
    ad_lt1 = delta_sharpe_ci(mret("A", False)[m_lt1.values], mret("D", False)[m_lt1.values])
    print(f"  ③ ΔSharpe(A−D) gross · **E<1 달 판정**: {ad_lt1['d']:+.4f} · "
          f"95% CI [{ad_lt1['lo']:+.4f}, {ad_lt1['hi']:+.4f}] · n={ad_lt1['n']} "
          f"→ {'✓' if (ad_lt1['n'] >= MIN_E_LT1_MONTHS and ad_lt1['d'] > 0) else '✗'}"
          f"{' (표본 부족)' if ad_lt1['n'] < MIN_E_LT1_MONTHS else ''}")
    print(f"     (전체 표본 병기: {ad_all['d']:+.4f}, CI [{ad_all['lo']:+.4f}, {ad_all['hi']:+.4f}], n={ad_all['n']})")
    ae_d = (mret("A") - mret("E")).dropna() * 100
    n = len(ae_d)
    se = ae_d.std(ddof=1) / np.sqrt(n)
    ae_s = delta_sharpe_ci(mret("A"), mret("E"))
    print(f"  ④ A−E 월초과 평균: {ae_d.mean():+.3f}%p · 중위 {ae_d.median():+.3f}%p · "
          f"t={ae_d.mean()/se:.2f} · CI [{ae_d.mean()-1.96*se:+.3f}, {ae_d.mean()+1.96*se:+.3f}] "
          f"→ {'✓' if ae_d.mean() > 0 else '✗'}")
    print(f"     ΔSharpe(A−E) 타이브레이커: {ae_s['d']:+.4f} "
          f"→ {'부호 일치' if (ae_s['d'] > 0) == (ae_d.mean() > 0) else '**부호 반대 → 통과로 세지 않음**'}")
    for lbl, g in (("A−C", "C"), ("A−E", "E")):
        d = (mret("A") - mret(g)).dropna()
        print(f"     [{lbl}] 음/영/양 달: {int((d<0).sum())}/{int((d==0).sum())}/{int((d>0).sum())}")

    s = rows["A"]
    print(f"\n[사전 판정 기준]")
    print(f"  ① A vs B 평균>0 & 중위>0 (비용후): 평균 {s['exc_mean']:+.3f}%p · "
          f"중위 {s['exc_med']:+.3f}%p → {'✓' if (s['exc_mean']>0 and s['exc_med']>0) else '✗'}")

    # 추가 채택 제한 (§3) — ②(전체표본 CI) 또는 ③(E<1 CI)이 0을 포함하면 최대 부분지지
    ci0 = (ac["lo"] <= 0 <= ac["hi"]) or (ad_lt1["lo"] <= 0 <= ad_lt1["hi"])
    print(f"  ▸ 추가 채택 제한(§3): ②·③ CI 중 0을 포함하는 것이 "
          f"{'있음 → **최대 부분 지지(registry 기각)**' if ci0 else '없음'}")

    inv_mask = invested_day_mask(len(P["dates"]), shadows["A"]["rebs"], bull)
    ev, warm, low_vol = [], 0, 0
    for i, v in sorted(exposures["A"].items()):
        ev.append(v)
        if len(np.flatnonzero(inv_mask[:i] & np.isfinite(shadows["A"]["daily_ret"][:i]))) < VOL_WIN:
            warm += 1
        elif v >= 1 - 1e-12:
            low_vol += 1
    ev = np.array(ev)
    n_lt1 = int((ev < 1 - 1e-12).sum())
    print(f"\n[부가] E 평균 {ev.mean():.3f} · 중위 {np.median(ev):.3f} · "
          f"E<1 {n_lt1}회 / 워밍업 E=1 {warm}회 / 저변동성 E=1 {low_vol}회 "
          f"(합 {n_lt1+warm+low_vol}={len(ev)})")
    # 적격 종목수 분포 — 가설의 "가변 breadth" 주장을 직접 확인
    cnt = [len(select(P, i, GROUP_SPEC["A"], cfg)[0])
           for i in shadows["A"]["rebs"] if bull[i]]
    print(f"[부가] 강세 리밸일 적격 종목수: 평균 {np.mean(cnt):.0f} · "
          f"최소 {min(cnt)} · 최대 {max(cnt)} (상한 {cfg['universe_n']})")
    print(f"[부가] 그룹별 **목표 대비 실현** 익스포저 · 목표비중 대비 실현비중 평균절대편차:", end=" ")
    for g in ("A", "C", "D", "E"):
        print(f"{g} {rows[g]['real_exp']*100:.1f}%/{rows[g]['w_dev']*1e4:.1f}bp", end="  ")
    # 민감도 2종 (§2.3·§3 사전 등록, 판정 미사용)
    nx = simulate(P, bull, "A", cfg, cfg["cost_pct"], exposures=exposures["A"],
                  exec_next_open=True)
    sn = stats(nx["equity"].reindex(a_eq.index).ffill(), bench_eq)
    print(f"\n[민감도] A 익일시가 체결: 월초과 평균 {sn['exc_mean']:+.3f}%p · "
          f"중위 {sn['exc_med']:+.3f}%p · CAGR {sn['cagr']:.2f}% · Sharpe {sn['sharpe']:.2f}")
    alt_ex = {}
    for g in ("A",):
        sh = shadows[g]
        inv = invested_day_mask(len(P["dates"]), sh["rebs"], bull)
        alt_ex[g] = exposure_series(sh["daily_ret"], inv, sh["rebs"], VOL_TARGET_ALT)
    alt = simulate(P, bull, "A", cfg, cfg["cost_pct"], exposures=alt_ex["A"])
    sa = stats(alt["equity"].reindex(a_eq.index).ffill(), bench_eq)
    print(f"[민감도] A 타깃 {VOL_TARGET_ALT*100:.0f}%: 월초과 평균 {sa['exc_mean']:+.3f}%p · "
          f"중위 {sa['exc_med']:+.3f}%p · CAGR {sa['cagr']:.2f}% · Sharpe {sa['sharpe']:.2f} · "
          f"E 평균 {np.mean([v for v in alt_ex['A'].values()]):.3f}")

    seg = np.array([bull_map.get(d, False) for d in P["dates"]
                    if a_eq.index[0].date() <= d <= a_eq.index[-1].date()])
    print(f"\n[레짐] 강세 거래일 {seg.mean()*100:.1f}% · "
          f"[좀비] A {rows['A']['zombie']}건 (리밸일 발생 건수 누적 — 고유 종목수 아님)")
    print(f"[구간] {a_eq.index[0].date()} ~ {a_eq.index[-1].date()} "
          f"(독립 표본 {rows['A']['n_months']}개월)")
    return rows, out, exposures, P, bull, a_eq, bench_eq


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest():
    # ① Memmel CI — 문서 §3의 극단값 검증을 코드로 재확인
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.01, 0.05, 400))
    same = delta_sharpe_ci(x, x.copy())
    assert abs(same["d"]) < 1e-12 and abs(same["hi"] - same["lo"]) < 1e-9, \
        f"동일 계열인데 SE≠0: {same}"
    z1 = pd.Series(rng.normal(0, 0.05, 5000))
    z2 = pd.Series(rng.normal(0, 0.05, 5000))
    ind = delta_sharpe_ci(z1, z2)
    want_se = np.sqrt(2 / 5000)
    got_se = (ind["hi"] - ind["lo"]) / (2 * 1.96)
    assert abs(got_se - want_se) / want_se < 0.15, f"독립 SE {got_se} vs {want_se}"

    # ② 투자일 마스크 — 리밸→리밸 블록
    bull = np.array([True, True, False, False, True, True, True, True])
    m = invested_day_mask(8, [1, 4, 6], bull)
    # 블록 (1,4]: bull[1]=True → 2,3,4 투자일 / (4,6]: bull[4]=True → 5,6 / (6,7]: bull[6]=True → 7
    assert list(np.flatnonzero(m)) == [2, 3, 4, 5, 6, 7], list(np.flatnonzero(m))
    bull2 = np.array([True, False, True, True, True, True])
    m2 = invested_day_mask(6, [1, 3], bull2)
    assert list(np.flatnonzero(m2)) == [4, 5], list(np.flatnonzero(m2))   # (1,3]은 약세라 제외

    # ③ 익스포저 — 변동성이 타깃보다 크면 E<1, 작으면 1
    dr = np.full(400, np.nan)
    dr[1:] = 0.02 * np.sign(np.arange(399) % 2 - 0.5)      # 일간 ±2% → 연 31.7%
    inv = np.ones(400, dtype=bool)
    ex = exposure_series(dr, inv, [300], 0.12)
    assert 0 < ex[300] < 1, ex
    dr2 = np.full(400, np.nan)
    dr2[1:] = 0.0001 * np.sign(np.arange(399) % 2 - 0.5)   # 거의 무변동
    assert exposure_series(dr2, inv, [300], 0.12)[300] == 1.0

    # ④ 시뮬레이터 — 단일 종목·항상 강세·E=1 → 목표 100% 투자
    _selftest_single()
    # ⑤ 다종목 — 느린 참조 구현과 최종 자본 대조
    _selftest_multi()
    print("selftest: 5개 항목 통과 (Memmel 극단값·투자일 블록·익스포저·단일종목·"
          "다종목 참조 대조)")


def _fixture(px: np.ndarray, k: int = 1) -> dict:
    n = len(px)
    c = np.tile(px[:, None], (1, k)).astype(float)
    if k > 1:
        c = c * (1 + 0.001 * np.arange(k))
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    return {"dates": np.asarray(pd.bdate_range("2015-01-01", periods=n).date),
            "tickers": np.array([f"T{j:02d}" for j in range(k)]),
            "close": c, "close_ff": c, "open": c, "ret": ret,
            "mom": np.full_like(c, 0.5), "sigma": np.full_like(c, 0.02),
            "dollar_vol": np.full_like(c, 1e9),
            "bars": np.tile(np.arange(1, n + 1)[:, None], (1, k)).astype(float) + MIN_BARS}


def _selftest_single():
    m = 400
    P = _fixture(np.full(m, 100.0))
    r = simulate(P, np.ones(m, dtype=bool), "D", {"universe_n": 1}, 0.0)
    # E=1, 단일 종목 → 자본 전액을 정수 주로 → 자본 불변(가격 불변)
    assert abs(r["equity"].iloc[-1] - START_EQUITY) < 1e-6, r["equity"].iloc[-1]
    assert r["real_exp"] > 0.999, r["real_exp"]      # 재배분으로 누수 회수


def _reference_sim(P, bull, cfg, cost_pct, group):
    """simulate()를 리스트·sorted()만으로 다시 구현한 참조본 (E=1 고정)."""
    spec = GROUP_SPEC[group]
    close, n_days, n_tk = P["close"], P["close"].shape[0], P["close"].shape[1]
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_MOM, 1)
    rebs = set(month_end_indices(P["dates"], start_i))
    shares, cash, last = [0.0] * n_tk, START_EQUITY, START_EQUITY
    tick = P["tickers"]
    for i in range(min(rebs), n_days):
        last = cash + sum(shares[j] * (P["close_ff"][i][j]
                                       if np.isfinite(P["close_ff"][i][j]) else 0.0)
                          for j in range(n_tk))
        if i not in rebs:
            continue
        px = close[i]
        ok = [j for j in range(n_tk)
              if np.isfinite(px[j]) and px[j] > 0 and P["bars"][i][j] >= MIN_BARS
              and np.isfinite(P["sigma"][i][j]) and P["sigma"][i][j] > 0
              and (not spec["mom"] or (np.isfinite(P["mom"][i][j]) and P["mom"][i][j] > 0))]
        if not bull[i] or not ok:
            for j in range(n_tk):
                if shares[j] > 0 and np.isfinite(px[j]) and px[j] > 0:
                    cash += shares[j] * px[j] * (1 - cost_pct / 100.0)
                    shares[j] = 0.0
            continue
        raw = [(1.0 / P["sigma"][i][j] if spec["invvol"] else 1.0) for j in ok]
        tot = sum(raw)
        w = {j: raw[t] / tot for t, j in enumerate(ok)}
        order = sorted(ok, key=lambda j: (-w[j], tick[j]))
        base = cash + sum(shares[j] * (P["close_ff"][i][j]
                                       if np.isfinite(P["close_ff"][i][j]) else 0.0)
                          for j in range(n_tk))
        q = {j: float(np.floor(base * w[j] / px[j])) for j in ok}
        for j in range(n_tk):
            if shares[j] <= 0 or not (np.isfinite(px[j]) and px[j] > 0):
                continue
            exc = shares[j] - q.get(j, 0.0)
            if exc > 0:
                cash += exc * px[j] * (1 - cost_pct / 100.0)
                shares[j] -= exc
        for j in order:
            if shares[j] >= q[j]:
                continue
            need = (q[j] - shares[j]) * px[j] * (1 + cost_pct / 100.0)
            if need > cash:
                continue
            cash -= need
            shares[j] = q[j]
        held = sum(shares[j] * px[j] for j in ok)
        budget = base - held
        for j in order:
            amt = px[j] * (1 + cost_pct / 100.0)
            if budget <= 0 or amt > budget or amt > cash:
                continue
            cash -= amt
            budget -= amt
            shares[j] += 1.0
    return last


def _selftest_multi():
    rng = np.random.default_rng(5)
    m, k = 420, 12
    px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, m))
    P = _fixture(px, k)
    P["sigma"] = np.tile((0.01 + 0.002 * np.arange(k))[None, :], (m, 1))
    bull = np.ones(m, dtype=bool)
    bull[300:340] = False
    cfg = {"universe_n": k}
    got = simulate(P, bull, "D", cfg, 0.1)
    want = _reference_sim(P, bull, cfg, 0.1, "D")
    assert abs(got["equity"].iloc[-1] - want) < 1e-6, \
        f"벡터화 {got['equity'].iloc[-1]:.6f} vs 참조 {want:.6f}"
    assert got["n_pos"] > 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if arg == "--selftest":
        selftest()
    else:
        run_market(arg)
