"""H-009 — Clenow 지수회귀 모멘텀 랭킹 (주간 리밸런싱 포트폴리오).

설계 문서: docs/analysis/backtests/2026-07-31-clenow-momentum-ranking.md
규칙·대조군·사전 판정 기준은 전부 그 문서 §2~§3을 따른다.

**btlib를 쓰지 않고 새로 짠 부분과 그 이유**
btlib.engine은 "이벤트 1건 = 거래 1건"인 단일 거래 평가기라, 자본·현금·보유
포지션을 시점 간에 이어가는 포트폴리오 전략을 표현할 수 없다. 그래서 이 스크립트에
포트폴리오 시뮬레이터를 새로 작성한다. 데이터 로딩(`btlib.loading`)·레짐
(`btlib.regime`)·거래비용 원천값(`btlib.costs.COST_PCT`, 왕복 → 편도 환산)은
btlib를 그대로 쓴다.

종가 0/음수 종목 제외는 `build_panel`이 직접 수행한다(`btlib.loading`은 순수
parquet 로더라 필터링하지 않는다).

신규 로직은 --selftest 로 검증한다: 정답을 아는 소형 입력 + 느린 루프 구현 대조
(회귀 점수, 다종목 랭킹·선택 루프, 리밸런싱 일자).

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/clenow_momentum_ranking.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/clenow_momentum_ranking.py kr
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/clenow_momentum_ranking.py us
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
REG_WIN = 90            # 로그가격 회귀 창
ANNUAL = 250            # 점수 연율화 지수 (Clenow 원본)
ATR_WIN = 20
SMA_STOCK = 100         # 종목 추세 필터
SMA_BENCH = 200         # 레짐 (PROTOCOL §2)
GAP_WIN, GAP_THR = 90, 0.15
DV_WIN = 60             # 거래대금 중앙값 창
TOP_PCT = 0.20          # 랭킹 상위 20%
RISK_FACTOR = 0.001     # 계좌평가액 대비 ATR 리스크
MIN_BARS = 253          # 종목별 최소 이력 (문서 §2.2 — rs_raw 요건에 맞춤)
REBAL_WEEKDAY = 2       # 수요일 (월=0)
SEED = 20260731
START_EQUITY = 1e8

# 비용은 btlib.costs(왕복)를 단일 소스로 삼고 편도로 환산한다.
MARKET_CFG = {
    "us": {"universe_n": 500, "cost_pct": costs.COST_PCT["US"] / 2},
    "kr": {"universe_n": 300, "cost_pct": costs.COST_PCT["KR"] / 2},
}
GROUPS = ["A", "C", "D", "E"]                      # B(벤치마크)는 시뮬레이션 불필요


# ── 벡터화 지표 (패널 = date × ticker) ───────────────────────────────────

def _win_sums(a: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """길이 n 창의 (Σy, Σk·y, 유효표본수). NaN은 0으로 채우고 개수로 유효성을 가린다.

    누적합 두 번으로 구한다 (창 길이와 무관하게 O(T)):
      S[t]  = Σ_{i<t} y_i,        TS[u] = Σ_{m<u} S[m]
      Σ_{k=0}^{n-1} k·y_{t-n+1+k} = (n−1)·S[t+1] − (TS[t+1] − TS[t-n+2])
    (종목마다 상장 시점이 달라 앞쪽에 NaN이 길게 있으므로, 단순 cumsum은 쓸 수 없다.)
    """
    v = np.isfinite(a)
    y0 = np.where(v, a, 0.0)
    T, K = a.shape
    z = np.zeros((1, K))
    S = np.vstack([z, np.cumsum(y0, axis=0)])          # S[t] = Σ_{i<t}
    TS = np.vstack([z, np.cumsum(S, axis=0)])          # TS[u] = Σ_{m<u} S[m]
    cnt = np.vstack([z, np.cumsum(v, axis=0)])

    t = np.arange(T)
    lo = t - n + 1
    ok = lo >= 0
    sum_y = np.full((T, K), np.nan)
    sum_ky = np.full((T, K), np.nan)
    n_ok = np.zeros((T, K))
    ti, li = t[ok], lo[ok]
    sum_y[ok] = S[ti + 1] - S[li]
    sum_ky[ok] = (n - 1) * S[ti + 1] - (TS[ti + 1] - TS[li + 1])
    n_ok[ok] = cnt[ti + 1] - cnt[li]
    return sum_y, sum_ky, n_ok


def regression_score(close: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """90일 ln(종가) OLS: 점수 = (exp(b)^250 − 1) × R².

    x = 0..89 고정이라 닫힌 형태로 계산한다.
      b = Sxy/Sxx,  R² = Sxy² / (Sxx·Syy),  Sxy = Σk·y − x̄·Σy
    """
    n = REG_WIN
    a = np.log(close.to_numpy(float))
    x = np.arange(n, dtype=float)
    sxx = float(((x - x.mean()) ** 2).sum())

    sum_y, sum_ky, n_ok = _win_sums(a, n)
    sum_y2, _, _ = _win_sums(a ** 2, n)
    full = n_ok == n

    with np.errstate(divide="ignore", invalid="ignore"):
        sxy = sum_ky - x.mean() * sum_y
        syy = sum_y2 - sum_y ** 2 / n
        b = sxy / sxx
        r2 = (sxy ** 2) / (sxx * syy)
        score = (np.exp(b) ** ANNUAL - 1.0) * r2
    return np.where(full, score, np.nan), np.where(full, b, np.nan)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> np.ndarray:
    h, l, c = (df.to_numpy(float) for df in (high, low, close))
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    tr = np.nanmax(np.stack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    tr = np.where(np.isfinite(h) & np.isfinite(l) & np.isfinite(prev), tr, np.nan)
    s, _, n_ok = _win_sums(tr, ATR_WIN)
    return np.where(n_ok == ATR_WIN, s / ATR_WIN, np.nan)


def rs_score(close: pd.DataFrame) -> np.ndarray:
    """대조군 C — 기존 사이트의 IBD식 RS (config.RS_WEIGHTS와 동일 정의)."""
    from pipeline import config
    out = 0.0
    for lb, w in config.RS_WEIGHTS.items():
        out = out + w * (close / close.shift(lb) - 1.0)
    return out.to_numpy(float)


def build_panel(market: str):
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "high", "low", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # 종가 0/음수는 데이터 이상 → NaN 처리 (btlib 로더와 같은 방침)
    bad = (close <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    score, _ = regression_score(close)
    gap_day = ((close / close.shift(1) - 1.0).abs() >= GAP_THR).astype(float)
    P = {
        "dates": np.asarray(idx),
        "tickers": np.asarray(cols),
        "close": close.to_numpy(float),
        "close_ff": close.ffill().to_numpy(float),   # 보유분 평가용(데이터 끊김 대비)
        "open": piv["open"].to_numpy(float),
        "score": score,
        "rs": rs_score(close),
        "atr": atr(piv["high"], piv["low"], close),
        "sma": close.rolling(SMA_STOCK, min_periods=SMA_STOCK).mean().to_numpy(float),
        "gap": gap_day.rolling(GAP_WIN, min_periods=1).max().to_numpy(float),
        "dollar_vol": (close * piv["volume"]).to_numpy(float),
        "bars": close.notna().cumsum().to_numpy(float),
    }
    return P


# ── 포트폴리오 시뮬레이터 (btlib 미보유 로직 — 신규 작성) ────────────────

def rebalance_dates(dates: np.ndarray, start_i: int) -> list[int]:
    """매주 수요일. 그날이 휴장이면 **직전 거래일**(그 이전 방향)로 당긴다.

    각 주의 달력상 수요일을 만들고, 거래일 달력에서 그 날짜 **이하**의 마지막
    거래일을 취한다. 연휴로 월~수가 모두 휴장이면 전주 마지막 거래일과 같아지므로
    중복을 제거한다(그 주는 리밸런싱을 건너뛴 것과 같다). 미래 방향으로 밀지 않아
    look-ahead가 생기지 않는다.
    """
    s = pd.to_datetime(dates)
    weds = pd.date_range(s[0] - pd.Timedelta(days=7), s[-1], freq="W-WED")
    pos = np.searchsorted(s.values, weds.values, side="right") - 1
    out = sorted({int(p) for p in pos if p >= start_i})
    return out


def simulate(P, bull: np.ndarray, group: str, cfg: dict, cost_pct: float,
             rng: np.random.Generator, exec_next_open: bool = False) -> dict:
    """일별 equity 곡선과 회전율을 반환. group: A|C|D|E (문서 §2.5)."""
    close, dates = P["close"], P["dates"]
    n_days, n_tk = close.shape
    score_mat = P["rs"] if group == "C" else P["score"]

    start_i = int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS))
    start_i = max(start_i, SMA_BENCH, REG_WIN, 1)
    rebs = rebalance_dates(dates, start_i)
    if not rebs:
        raise RuntimeError("리밸런싱 일자 없음")

    shares = np.zeros(n_tk)
    cash = START_EQUITY
    equity = np.full(n_days, np.nan)
    invested = np.full(n_days, np.nan)      # 투자비중(1−현금비중) — 해석용 부가 기록
    n_pos = np.full(n_days, np.nan)
    traded_value = 0.0
    reb_set = set(rebs)

    close_ff = P["close_ff"]
    for i in range(rebs[0], n_days):
        px_i = close[i]
        held = shares > 0
        # 평가는 ffill 가격으로 한다 — 당일 데이터가 없는 보유 종목을 0으로 세면
        # 자본이 사라졌다가 되살아난다.
        val = float((shares * np.nan_to_num(close_ff[i])).sum())
        equity[i] = cash + val
        invested[i] = val / equity[i] if equity[i] > 0 else np.nan
        n_pos[i] = int(held.sum())

        if i not in reb_set:
            continue

        eq = equity[i]
        exec_px = P["open"][i + 1] if (exec_next_open and i + 1 < n_days) else px_i

        # 이번 시점 지표 스냅샷
        sc, atr_i, sma_i, gap_i = score_mat[i], P["atr"][i], P["sma"][i], P["gap"][i]
        valid = np.isfinite(px_i) & np.isfinite(exec_px) & (P["bars"][i] >= MIN_BARS)

        # 유동성 유니버스 (PIT — 직전 DV_WIN봉 거래대금 중앙값 상위 N)
        dv_slice = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
        with np.errstate(all="ignore"):
            dv_med = np.nanmedian(dv_slice, axis=0)
        univ = valid & np.isfinite(dv_med) & np.isfinite(sc)
        if univ.sum() == 0:
            continue
        n_keep = min(int(cfg["universe_n"]), int(univ.sum()))
        thr_dv = np.sort(dv_med[univ])[::-1][n_keep - 1]
        univ &= (dv_med >= thr_dv)

        # 랭킹 상위 20% — 모집단은 유동성 유니버스 전체 (문서 §2.2 용어 정의)
        top = np.zeros(n_tk, dtype=bool)
        u_idx = np.flatnonzero(univ)
        k = max(1, int(round(len(u_idx) * TOP_PCT)))
        top[u_idx[np.argsort(sc[u_idx])[::-1][:k]]] = True

        eligible = univ & (px_i > sma_i) & (gap_i == 0) & np.isfinite(atr_i) & (atr_i > 0)

        # ① 청산 — 랭킹 이탈(유니버스 탈락 포함) / SMA100 하회 / 갭 (E는 랭킹만)
        sell = held & ~top
        if group != "E":
            sell |= held & (~np.isfinite(px_i) | (px_i <= sma_i) | (gap_i == 1))
        sell &= np.isfinite(exec_px)
        if sell.any():
            gross = (shares[sell] * exec_px[sell]).sum()
            cash += gross * (1 - cost_pct / 100.0)
            traded_value += gross
            shares[sell] = 0.0
            held = shares > 0

        # ② 레짐: 약세면 신규 매수 없음
        if not bull[i]:
            continue

        # ③ 신규 매수 — 미보유 & 적격 & (A/C/E: 랭킹 상위 / D: 무작위 동수)
        cand = np.flatnonzero(eligible & ~held)
        if len(cand) == 0:
            continue
        ranked = cand[np.argsort(sc[cand])[::-1]]
        a_like = ranked[top[ranked]]
        if group == "D":
            m = min(len(a_like), len(cand))
            order = rng.choice(cand, size=m, replace=False) if m else np.array([], dtype=int)
        else:
            order = a_like

        for j in order:
            qty = np.floor(eq * RISK_FACTOR / atr_i[j])
            if qty <= 0:
                continue
            need = qty * exec_px[j] * (1 + cost_pct / 100.0)
            if need > cash:
                break                       # 하드 스톱 (문서 §2.3-3)
            cash -= need
            traded_value += qty * exec_px[j]
            shares[j] = qty

    idx_all = pd.to_datetime(dates)
    equity = pd.Series(equity, index=idx_all).dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return {"equity": equity, "turnover": traded_value / START_EQUITY / years,
            "invested": float(np.nanmean(invested)), "n_pos": float(np.nanmean(n_pos))}


# ── 성과 집계 ────────────────────────────────────────────────────────────

def monthly(series: pd.Series) -> pd.Series:
    return series.resample("ME").last().pct_change().dropna()


def stats(eq: pd.Series, bench_eq: pd.Series) -> dict:
    m, bm = monthly(eq), monthly(bench_eq)
    j = pd.concat([m, bm], axis=1, join="inner").dropna()
    exc = (j.iloc[:, 0] - j.iloc[:, 1]) * 100
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min() * 100
    r = m
    return {"n_months": len(exc), "exc_mean": exc.mean(), "exc_med": exc.median(),
            "cagr": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
            "vol": r.std() * np.sqrt(12) * 100,
            "sharpe": (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else float("nan"),
            "mdd": dd, "monthly": m}


def run_market(market: str):
    cfg = MARKET_CFG[market]
    P = build_panel(market)
    bench = loading.load_bench(market)
    bull_map = regime.bull_map(bench, sma=SMA_BENCH)
    bull = np.array([bull_map.get(d, False) for d in P["dates"]])
    bser = bench.set_index(pd.to_datetime(bench["date"]))["close"].astype(float)

    print(f"\n{'='*78}\n[{market.upper()}] 유니버스 {len(P['tickers']):,}종목 · "
          f"데이터 기준일 {P['dates'][-1]} · 상위 {cfg['universe_n']} · "
          f"편도비용 {cfg['cost_pct']}%\n{'='*78}")

    out = {}
    for net in (True, False):
        for g in GROUPS:
            rng = np.random.default_rng(SEED)
            out[(g, net)] = simulate(P, bull, g, cfg, cfg["cost_pct"] if net else 0.0, rng)

    a_eq = out[("A", True)]["equity"]
    bench_eq = bser.reindex(a_eq.index).ffill().dropna()
    a_eq = a_eq.reindex(bench_eq.index).dropna()
    bench_eq = bench_eq.reindex(a_eq.index)

    rows = {}
    for g in GROUPS:
        eq = out[(g, True)]["equity"].reindex(a_eq.index).ffill()
        rows[g] = stats(eq, bench_eq)
        for k in ("turnover", "invested", "n_pos"):
            rows[g][k] = out[(g, True)][k]
    rows["B"] = stats(bench_eq, bench_eq)
    rows["B"].update(turnover=0.0, invested=1.0, n_pos=float("nan"))

    print(f"{'그룹':<10} {'월수':>4} {'월초과평균':>9} {'월초과중위':>9} {'CAGR':>7} "
          f"{'변동성':>6} {'Sharpe':>6} {'MDD':>7} {'회전율':>7} {'투자비중':>7} {'종목수':>6}")
    print("-" * 96)
    label = {"A": "A 전략", "B": "B 벤치마크", "C": "C RS랭킹",
             "D": "D 무작위", "E": "E 청산축소"}
    for g in ["A", "B", "C", "D", "E"]:
        s = rows[g]
        print(f"{label[g]:<10} {s['n_months']:>4} {s['exc_mean']:>8.3f}% {s['exc_med']:>8.3f}% "
              f"{s['cagr']:>6.2f}% {s['vol']:>5.1f}% {s['sharpe']:>6.2f} "
              f"{s['mdd']:>6.1f}% {s['turnover']:>6.2f}x {s['invested']*100:>6.1f}% "
              f"{s['n_pos']:>5.1f}")

    # 그룹 간 차분 (월별 수익률 차이)
    def diff(g1, g2, net=True):
        e1 = out[(g1, net)]["equity"].reindex(a_eq.index).ffill()
        e2 = out[(g2, net)]["equity"].reindex(a_eq.index).ffill()
        d = (monthly(e1) - monthly(e2)).dropna() * 100
        return d.mean(), d.median()

    print("\n[핵심 차분 — 월별 수익률 차, %p]")
    for lbl, g1, g2, net in [("A − C (측정 방식)", "A", "C", True),
                             ("A − D net", "A", "D", True),
                             ("A − D gross ★판정", "A", "D", False),
                             ("A − E (청산조건 기여)", "A", "E", True)]:
        mu, md = diff(g1, g2, net)
        print(f"  {lbl:<22} 평균 {mu:>7.3f}%p · 중위 {md:>7.3f}%p")

    mu_ab, md_ab = rows["A"]["exc_mean"], rows["A"]["exc_med"]
    print(f"\n[사전 판정 기준 대조]")
    print(f"  ① A vs B 월초과 평균>0 & 중위>0 (비용후): "
          f"평균 {mu_ab:+.3f}%p {'✓' if mu_ab > 0 else '✗'} · "
          f"중위 {md_ab:+.3f}%p {'✓' if md_ab > 0 else '✗'}")
    mu_ac, md_ac = diff("A", "C")
    print(f"  ③ A−C 평균>0: {mu_ac:+.3f}%p {'✓' if mu_ac > 0 else '✗'} (중위 {md_ac:+.3f}%p)")
    mu_ad, md_ad = diff("A", "D", net=False)
    print(f"  ④ A−D gross 평균>0: {mu_ad:+.3f}%p {'✓' if mu_ad > 0 else '✗'} (중위 {md_ad:+.3f}%p)")

    # 민감도: 익일 시가 체결
    rng = np.random.default_rng(SEED)
    nx = simulate(P, bull, "A", cfg, cfg["cost_pct"], rng, exec_next_open=True)
    sn = stats(nx["equity"].reindex(a_eq.index).ffill(), bench_eq)
    print(f"\n[민감도] A 익일시가 체결: 월초과 평균 {sn['exc_mean']:+.3f}%p · "
          f"중위 {sn['exc_med']:+.3f}%p · CAGR {sn['cagr']:.2f}%")
    print(f"[구간] {a_eq.index[0].date()} ~ {a_eq.index[-1].date()} "
          f"({rows['A']['n_months']}개월)")
    # 레짐 노출 — 백테스트 구간 중 벤치마크가 SMA200 위였던 거래일 비율.
    # 투자비중 차이를 설명하는 값이라 결과 문서에서 인용하므로 여기서 함께 출력한다.
    seg = np.array([bull_map.get(d, False) for d in P["dates"]
                    if a_eq.index[0].date() <= d <= a_eq.index[-1].date()])
    print(f"[레짐] 강세(지수>SMA200) 거래일 비율 {seg.mean()*100:.1f}% "
          f"({int(seg.sum()):,}/{len(seg):,}일)")
    return rows


# ── 자체 검증 (신규 로직 — 합성 입력 + 느린 루프 대조) ────────────────────

def selftest():
    rng = np.random.default_rng(0)
    n = 300
    # ① 회귀 점수: 정확히 지수 성장하는 계열은 R²=1, 기울기 b가 알려진 값
    g = 0.001
    close = pd.DataFrame({"T": 100 * np.exp(g * np.arange(n))})
    sc, b = regression_score(close)
    assert abs(b[-1] - g) < 1e-9, f"기울기 {b[-1]} != {g}"
    assert abs(sc[-1] - ((np.exp(g) ** ANNUAL - 1))) < 1e-6, "R²=1이면 점수=연율수익률"

    # ② 잡음 계열: 느린 루프 OLS와 대조
    y = pd.DataFrame({"T": 100 * np.cumprod(1 + rng.normal(0, 0.02, n))})
    sc2, b2 = regression_score(y)
    ly = np.log(y["T"].to_numpy())
    for t in (REG_WIN - 1, 150, n - 1):
        w = ly[t - REG_WIN + 1:t + 1]
        x = np.arange(REG_WIN, dtype=float)
        bb, aa = np.polyfit(x, w, 1)
        rr = np.corrcoef(x, w)[0, 1] ** 2
        want = (np.exp(bb) ** ANNUAL - 1) * rr
        assert abs(b2[t] - bb) < 1e-9, f"t={t} 기울기 {b2[t]} vs {bb}"
        assert abs(sc2[t] - want) < 1e-9, f"t={t} 점수 {sc2[t]} vs {want}"

    # ③ 시뮬레이터: 단조 상승 단일 종목 + 항상 강세 → 매수 후 보유와 같아야 함
    m = 400
    px = np.linspace(100.0, 300.0, m)
    dates = pd.bdate_range("2015-01-01", periods=m).date
    P = {"dates": np.asarray(dates), "tickers": np.array(["T"]),
         "close": px[:, None], "close_ff": px[:, None], "open": px[:, None],
         "score": np.ones((m, 1)), "rs": np.ones((m, 1)),
         "atr": np.full((m, 1), 1.0), "sma": (px - 10)[:, None],
         "gap": np.zeros((m, 1)), "dollar_vol": np.full((m, 1), 1e9),
         "bars": np.arange(1, m + 1)[:, None].astype(float)}
    r = simulate(P, np.ones(m, dtype=bool), "A", {"universe_n": 1}, 0.0,
                 np.random.default_rng(0))
    eq = r["equity"]
    qty = np.floor(START_EQUITY * RISK_FACTOR / 1.0)
    i0 = eq.index[0]
    i0_pos = list(pd.to_datetime(P["dates"])).index(i0)
    want_end = START_EQUITY + qty * (px[-1] - px[i0_pos])
    assert abs(eq.iloc[-1] - want_end) < 1e-6, f"{eq.iloc[-1]} vs {want_end}"

    # ④ 레짐 False면 매수가 없어 자본 불변
    r2 = simulate(P, np.zeros(m, dtype=bool), "A", {"universe_n": 1}, 0.0,
                  np.random.default_rng(0))
    assert abs(r2["equity"].iloc[-1] - START_EQUITY) < 1e-9

    # ⑤ 비용이 붙으면 자본이 줄어든다 (동일 조건 비교)
    r3 = simulate(P, np.ones(m, dtype=bool), "A", {"universe_n": 1}, 0.5,
                  np.random.default_rng(0))
    assert r3["equity"].iloc[-1] < eq.iloc[-1]

    # ⑥ 리밸런싱 일자: 수요일 이하로만 당겨지는가 (연휴 주 포함)
    #    2026-02-16(월)~18(수) 휴장을 가정한 거래일 달력을 만들어 확인한다.
    cal = pd.bdate_range("2026-02-02", "2026-03-06")
    cal = cal[~cal.isin(pd.to_datetime(["2026-02-16", "2026-02-17", "2026-02-18"]))]
    rb = rebalance_dates(np.asarray(cal.date), 0)
    picked = pd.to_datetime([cal.date[i] for i in rb])
    # 휴장이 없는 주는 수요일이 그대로 선택된다
    for d in ("2026-02-04", "2026-02-11", "2026-02-25", "2026-03-04"):
        assert pd.Timestamp(d) in picked, f"{d}(수) 누락"
    # 수요일(2/18)이 휴장인 주는 **직전 거래일**(2/13 금)로 당겨진다
    assert pd.Timestamp("2026-02-13") in picked, "휴장 주가 직전 거래일로 당겨지지 않음"
    # 미래 방향(2/19 목·2/20 금)으로는 절대 밀리지 않는다 — look-ahead 방지
    assert not ({pd.Timestamp("2026-02-19"), pd.Timestamp("2026-02-20")} & set(picked)), \
        "연휴 주에 미래로 밀렸다"
    assert set(picked) <= set(pd.to_datetime(cal.date)), "거래일이 아닌 날짜 선택"

    # ⑦ 다종목 랭킹·선택 루프 — 느린 참조 구현과 최종 자본 대조
    _selftest_multi()

    print("selftest: 7개 항목 통과 (회귀 닫힌형·루프 대조·매수후보유 등가·레짐·"
          "비용·리밸일·다종목 랭킹 루프)")


def _reference_sim(P, bull, cost_pct, cfg, rebs):
    """simulate()의 A 그룹을 '읽기 쉬운 느린 방식'으로 다시 구현한 참조본.

    벡터화·부분정렬 없이 파이썬 리스트와 sorted()만 쓴다. 두 구현의 최종 자본이
    일치하면 랭킹 임계값·상위 20% 선정·하드스톱 루프가 같다는 뜻이다.
    """
    close, n_days, n_tk = P["close"], P["close"].shape[0], P["close"].shape[1]
    shares, cash = [0.0] * n_tk, START_EQUITY
    last_eq = START_EQUITY
    for i in range(rebs[0], n_days):
        val = sum(shares[j] * (P["close_ff"][i][j] if np.isfinite(P["close_ff"][i][j]) else 0.0)
                  for j in range(n_tk))
        last_eq = cash + val
        if i not in rebs:
            continue
        eq = last_eq
        # 유동성 유니버스
        univ = []
        for j in range(n_tk):
            col = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1, j]
            col = [v for v in col if np.isfinite(v)]
            if (not col or not np.isfinite(close[i][j]) or not np.isfinite(P["score"][i][j])
                    or P["bars"][i][j] < MIN_BARS):
                continue
            univ.append((float(np.median(col)), j))
        if not univ:
            continue
        univ.sort(key=lambda x: -x[0])
        keep = min(cfg["universe_n"], len(univ))
        thr = univ[keep - 1][0]
        u = [j for dv, j in univ if dv >= thr]
        k = max(1, int(round(len(u) * TOP_PCT)))
        top = {j for j in sorted(u, key=lambda j: -P["score"][i][j])[:k]}
        # 청산
        for j in range(n_tk):
            if shares[j] <= 0 or not np.isfinite(close[i][j]):
                continue
            if (j not in top) or close[i][j] <= P["sma"][i][j] or P["gap"][i][j] == 1:
                cash += shares[j] * close[i][j] * (1 - cost_pct / 100.0)
                shares[j] = 0.0
        if not bull[i]:
            continue
        # 매수
        elig = [j for j in u if j in top and shares[j] <= 0
                and np.isfinite(close[i][j]) and close[i][j] > P["sma"][i][j]
                and P["gap"][i][j] == 0 and np.isfinite(P["atr"][i][j]) and P["atr"][i][j] > 0]
        for j in sorted(elig, key=lambda j: -P["score"][i][j]):
            qty = float(np.floor(eq * RISK_FACTOR / P["atr"][i][j]))
            if qty <= 0:
                continue
            need = qty * close[i][j] * (1 + cost_pct / 100.0)
            if need > cash:
                break
            cash -= need
            shares[j] = qty
    return last_eq


def _selftest_multi():
    rng = np.random.default_rng(7)
    m, k = 320, 12
    dates = pd.bdate_range("2016-01-01", periods=m).date
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.02, (m, k)), axis=0)
    close[:40, 3] = np.nan                       # 늦게 상장한 종목
    df = pd.DataFrame(close, index=pd.to_datetime(dates))
    sc, _ = regression_score(df)
    P = {"dates": np.asarray(dates), "tickers": np.arange(k).astype(str),
         "close": close, "close_ff": df.ffill().to_numpy(float), "open": close,
         "score": sc, "rs": sc,
         "atr": np.abs(df.diff()).rolling(ATR_WIN).mean().to_numpy(float),
         "sma": df.rolling(SMA_STOCK, min_periods=SMA_STOCK).mean().to_numpy(float),
         "gap": ((df / df.shift(1) - 1).abs() >= GAP_THR).astype(float)
                .rolling(GAP_WIN, min_periods=1).max().to_numpy(float),
         "dollar_vol": close * 1e6,
         "bars": df.notna().cumsum().to_numpy(float)}
    bull = np.ones(m, dtype=bool)
    bull[200:240] = False                        # 약세 구간 포함
    cfg = {"universe_n": 8}

    got = simulate(P, bull, "A", cfg, 0.1, np.random.default_rng(0))
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, REG_WIN, 1)
    rebs = set(rebalance_dates(P["dates"], start_i))
    want = _reference_sim(P, bull, 0.1, cfg, sorted(rebs))
    assert abs(got["equity"].iloc[-1] - want) < 1e-6, \
        f"벡터화 {got['equity'].iloc[-1]:.6f} vs 참조 루프 {want:.6f}"
    assert got["n_pos"] > 0, "매수가 한 건도 없으면 대조 의미가 없다"


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if arg == "--selftest":
        selftest()
    else:
        run_market(arg)
