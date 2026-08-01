"""H-010 — 정보 이산성(FIP) 필터 모멘텀 (월간 리밸런싱 동일비중 포트폴리오).

설계 문서: docs/analysis/backtests/2026-08-01-fip-information-discreteness.md
규칙·대조군·사전 판정 기준은 전부 그 문서 §2~§3을 따른다.

**btlib를 쓰지 않고 새로 짠 부분과 그 이유**
btlib.engine은 "이벤트 1건 = 거래 1건"인 단일 거래 평가기라 포트폴리오를 표현할 수
없다. H-009가 만든 시뮬레이터는 주간·ATR 사이징·재조정 없음 구조라 이 가설(월간·
동일비중·매월 재조정)과 매매 절차가 달라 그대로 쓸 수 없다. 그래서 월간 동일비중
시뮬레이터를 새로 작성한다. 데이터 로딩(`btlib.loading`)·레짐(`btlib.regime`)·
비용 원천값(`btlib.costs`)은 btlib을 그대로 쓴다.

종가 0/음수는 **해당 봉만 NaN**으로 마스킹한다(종목 통째 배제는 look-ahead —
문서 §2.2). btlib.engine의 `np.all(c > 0)` 방식과 의도적으로 다르다.

신규 로직은 --selftest 로 검증한다(정답을 아는 소형 입력 + 느린 참조 구현 대조).

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/fip_information_discreteness.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/fip_information_discreteness.py kr
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/fip_information_discreteness.py us
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
LB_FAR, LB_NEAR = 252, 21       # PRET = close[t-21]/close[t-252] - 1
N_RET = LB_FAR - LB_NEAR        # 231 — 구간 내 일간수익률 개수(분모 고정)
MIN_VALID_RET = 220             # 유효 수익률일 하한 (문서 §2.1)
MIN_BARS = LB_FAR + 1           # 253
DV_WIN = 60                     # 거래대금 중앙값 창 (판단일 포함)
POOL_N = 90                     # 적격 후보 고정 개수 (양 시장 공통)
HOLD_N = 20                     # 선정 종목 수
SMA_BENCH = 200
SEED = 20260801
SEEDS_D = (20260801, 20260802, 20260803)
START_EQUITY = 1e8

MARKET_CFG = {
    "us": {"universe_n": 500, "cost_pct": costs.COST_PCT["US"] / 2},   # 편도
    "kr": {"universe_n": 300, "cost_pct": costs.COST_PCT["KR"] / 2},
}
GROUPS = ["A", "C", "D", "E"]      # B(벤치마크)는 시뮬레이션 불필요


# ── 지표 (패널 = date × ticker, 공통 거래일 축) ──────────────────────────

def _count_window(mask: np.ndarray, n: int) -> np.ndarray:
    """길이 n 창의 True 개수. 종목마다 상장 시점이 달라 단순 cumsum은 못 쓴다."""
    T, K = mask.shape
    cs = np.vstack([np.zeros((1, K)), np.cumsum(mask.astype(float), axis=0)])
    out = np.full((T, K), np.nan)
    t = np.arange(T)
    ok = t - n + 1 >= 0
    out[ok] = cs[t[ok] + 1] - cs[t[ok] - n + 1]
    return out


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # 종가/시가 0·음수는 해당 봉만 NaN (종목 배제 금지 — 문서 §2.2).
    # KR 캐시의 거래정지 봉은 open=high=low=0, volume=0이고 close만 직전가로 이월돼
    # 있다(9,008봉). close만 보고 마스킹하면 0이 유효 체결가로 쓰여 floor(금액/0)=inf
    # → need=NaN → 가드 통과 → 현금 NaN → equity 조용한 절단으로 이어진다.
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    # PRET — 공통 축 위치 기준, ffill 없음
    far, near = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[LB_FAR:] = c[:-LB_FAR]
    near[LB_NEAR:] = c[:-LB_NEAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        pret = near / far - 1.0

    # %pos / %neg — 구간 t-252..t-21의 일간수익률 231개, 분모 231 고정
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    pos = _count_window(np.isfinite(ret) & (ret > 0), N_RET)
    neg = _count_window(np.isfinite(ret) & (ret < 0), N_RET)
    valid = _count_window(np.isfinite(ret), N_RET)
    # 창을 t-21까지로 당긴다 (판단일 t 기준 t-252..t-21 구간)
    def shift_back(a):
        out = np.full_like(a, np.nan)
        out[LB_NEAR:] = a[:-LB_NEAR]
        return out
    pos, neg, valid = shift_back(pos), shift_back(neg), shift_back(valid)

    with np.errstate(invalid="ignore"):
        idv = np.sign(pret) * ((neg - pos) / N_RET)
    idv = np.where(valid >= MIN_VALID_RET, idv, np.nan)

    return {
        "dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
        "close": c, "close_ff": close.ffill().to_numpy(float),
        "open": piv["open"].to_numpy(float),
        "pret": pret, "id": idv,
        "dollar_vol": (close * piv["volume"]).to_numpy(float),
        "bars": close.notna().cumsum().to_numpy(float),
    }


# ── 월간 동일비중 시뮬레이터 (신규 작성) ─────────────────────────────────

def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    """각 월의 마지막 거래일 인덱스."""
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).max() if v >= start_i]


def count_boundary_ties(P, i: int, sel: np.ndarray, group: str, pool: np.ndarray) -> int:
    """20위 경계에서 동점(같은 정렬 키)으로 결정된 슬롯 수 — §3 사전 등록 부가 기록."""
    if group == "D" or len(sel) < HOLD_N or len(pool) <= HOLD_N:
        return 0
    key = {"A": P["id"][i], "C": -P["pret"][i], "E": -P["id"][i]}[group]
    cut = key[sel[-1]]                      # 20위의 키 값
    n_in = int(np.sum(key[sel] == cut))     # 선정된 것 중 그 값과 같은 개수
    n_all = int(np.sum(key[pool] == cut))   # 후보 전체 중 같은 값 개수
    return n_in if n_all > n_in else 0      # 잘린 동점이 있을 때만 카운트


def select(P, i: int, group: str, cfg: dict, rng, ties_out: list | None = None) -> np.ndarray:
    """문서 §2.2~§2.3의 목표 포트폴리오 산출. 반환: 종목 인덱스 배열(집행 순서)."""
    n_tk = P["close"].shape[1]
    px_i = P["close"][i]
    base = (np.isfinite(px_i) & (P["bars"][i] >= MIN_BARS)
            & np.isfinite(P["pret"][i]) & np.isfinite(P["id"][i]))
    if not base.any():
        return np.array([], dtype=int)

    # 유동성 유니버스 — 판단일 포함 직전 60봉 거래대금 중앙값 상위 N (PIT)
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dv_med = np.nanmedian(dv, axis=0)
    univ = base & np.isfinite(dv_med)
    if not univ.any():
        return np.array([], dtype=int)
    keep = min(int(cfg["universe_n"]), int(univ.sum()))
    thr = np.sort(dv_med[univ])[::-1][keep - 1]
    univ &= dv_med >= thr

    # 적격 후보 — PRET>0 상위 90종목 (동점은 티커 오름차순)
    cand = np.flatnonzero(univ & (P["pret"][i] > 0))
    if len(cand) == 0:
        return np.array([], dtype=int)
    tick = P["tickers"]
    order = sorted(cand, key=lambda j: (-P["pret"][i][j], tick[j]))
    pool = np.array(order[:POOL_N], dtype=int)

    if group == "D":
        m = min(HOLD_N, len(pool))
        return np.sort(rng.choice(pool, size=m, replace=False))
    key = {"A": lambda j: (P["id"][i][j], tick[j]),        # ID 오름차순
           "C": lambda j: (-P["pret"][i][j], tick[j]),     # PRET 내림차순
           "E": lambda j: (-P["id"][i][j], tick[j])}[group]  # ID 내림차순
    sel = np.array(sorted(pool, key=key)[:HOLD_N], dtype=int)
    if ties_out is not None:
        ties_out.append(count_boundary_ties(P, i, sel, group, pool))
    return sel


def simulate(P, bull, group, cfg, cost_pct, rng, exec_next_open=False) -> dict:
    close, dates = P["close"], P["dates"]
    n_days, n_tk = close.shape
    close_ff = P["close_ff"]

    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_FAR, 1)
    rebs = month_end_indices(dates, start_i)
    if not rebs:
        raise RuntimeError("리밸런싱 일자 없음")
    reb_set = set(rebs)

    shares = np.zeros(n_tk)
    cash = START_EQUITY
    equity = np.full(n_days, np.nan)
    invested, n_pos = np.full(n_days, np.nan), np.full(n_days, np.nan)
    traded, zombie, ties = 0.0, 0, []
    eq_sum = 0.0          # 회전율 분모용 — 평균 자본 (초기자본 고정은 복리 배수가 섞인다)

    for i in range(rebs[0], n_days):
        val = float((shares * np.nan_to_num(close_ff[i])).sum())
        equity[i] = cash + val
        eq_sum += equity[i]
        invested[i] = val / equity[i] if equity[i] > 0 else np.nan
        n_pos[i] = int((shares > 0).sum())
        if i not in reb_set:
            continue

        exec_px = P["open"][i + 1] if (exec_next_open and i + 1 < n_days) else close[i]
        tradable = np.isfinite(exec_px) & (exec_px > 0)   # 0은 거래정지 placeholder
        zombie += int(((shares > 0) & ~tradable).sum())

        target = (np.array([], dtype=int) if not bull[i]
                  else select(P, i, group, cfg, rng, ties_out=ties))
        tset = set(int(j) for j in target)

        # ① 목표에 없는 보유분 전량 매도 (매도 선행)
        for j in np.flatnonzero(shares > 0):
            if int(j) in tset or not tradable[j]:
                continue
            cash += shares[j] * exec_px[j] * (1 - cost_pct / 100.0)
            traded += shares[j] * exec_px[j]
            shares[j] = 0.0
        if len(target) == 0:
            continue

        # ② 기준 자산 = 매도 후 (현금 + 보유 평가액), 종목당 목표 = 1/HOLD_N
        eq = cash + float((shares * np.nan_to_num(close_ff[i])).sum())
        tgt_amt = eq / HOLD_N

        # ③ 부분 매도를 모두 집행한 뒤 매수 (매도 대금이 매수 재원)
        for j in target:
            if shares[j] <= 0 or not tradable[j]:
                continue
            cur = shares[j] * exec_px[j]
            if cur - tgt_amt >= exec_px[j]:          # 차액이 1주 미만이면 거래 안 함
                q = np.floor((cur - tgt_amt) / exec_px[j])
                cash += q * exec_px[j] * (1 - cost_pct / 100.0)
                traded += q * exec_px[j]
                shares[j] -= q
        for j in target:                              # 집행 순서 = select() 반환 순서
            if not tradable[j]:
                continue
            cur = shares[j] * exec_px[j]
            if tgt_amt - cur < exec_px[j]:
                continue
            q = np.floor((tgt_amt - cur) / exec_px[j])
            need = q * exec_px[j] * (1 + cost_pct / 100.0)
            if q <= 0 or need > cash:
                continue                              # 현금 부족은 건너뛰고 다음 종목
            cash -= need
            traded += q * exec_px[j]
            shares[j] += q

    eq_s = pd.Series(equity, index=pd.to_datetime(dates)).dropna()
    # 조용한 절단 감지 — inf/NaN이 현금에 새어 들어가면 dropna로 시리즈가 잘린다.
    # 이번 검토에서 실제로 발생했던 실패 모드라 예외로 승격한다.
    if len(eq_s) != n_days - rebs[0]:
        raise RuntimeError(
            f"equity 시리즈 절단: {len(eq_s)}행 (기대 {n_days - rebs[0]}행). "
            f"마지막 유효일 {eq_s.index[-1].date() if len(eq_s) else '없음'}")
    n_obs = max(len(eq_s), 1)
    years = max((eq_s.index[-1] - eq_s.index[0]).days / 365.25, 1e-9)
    avg_eq = eq_sum / n_obs
    return {"equity": eq_s,
            # 단방향 회전율(매수+매도 합의 절반)을 평균 자본으로 나눈다
            "turnover": (traded / 2.0) / avg_eq / years,
            "invested": float(np.nanmean(invested)), "n_pos": float(np.nanmean(n_pos)),
            "zombie": zombie, "ties": float(np.mean(ties)) if ties else 0.0}


def simulate_overlap(P, bull, cfg, cost_pct, n_tranche: int = 6) -> dict:
    """6개월 중첩 트랑슈 — 논문(Da·Gurun·Warachka 2014)의 overlapping portfolio 세팅.

    매월 자본의 1/6로 A 규칙의 20종목을 새로 뽑고, 각 트랑슈를 6개월 뒤 청산한다.
    §3에 사전 등록된 민감도이며 판정에는 쓰지 않는다.
    """
    close, dates = P["close"], P["dates"]
    n_days, n_tk = close.shape
    close_ff = P["close_ff"]
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_FAR, 1)
    rebs = month_end_indices(dates, start_i)
    reb_set = set(rebs)
    tranches: list[dict] = []            # {shares: ndarray, die: 리밸 회차}
    cash = START_EQUITY
    equity = np.full(n_days, np.nan)
    reb_no = 0

    for i in range(rebs[0], n_days):
        held = sum(float((t["shares"] * np.nan_to_num(close_ff[i])).sum()) for t in tranches)
        equity[i] = cash + held
        if i not in reb_set:
            continue
        reb_no += 1
        px = close[i]
        tradable = np.isfinite(px) & (px > 0)

        # 만기 트랑슈 청산 + 레짐 약세면 전량 청산
        keep = []
        for t in tranches:
            if t["die"] <= reb_no or not bull[i]:
                sh = np.where(tradable, t["shares"], 0.0)
                cash += float((sh * np.nan_to_num(px)).sum()) * (1 - cost_pct / 100.0)
                rest = t["shares"] - sh
                if rest.sum() > 0:
                    keep.append({"shares": rest, "die": reb_no + 1})
            else:
                keep.append(t)
        tranches = keep
        if not bull[i]:
            continue

        # 새 트랑슈 = 현재 자본의 1/6
        eq = cash + sum(float((t["shares"] * np.nan_to_num(close_ff[i])).sum())
                        for t in tranches)
        budget = eq / n_tranche
        sel = select(P, i, "A", cfg, np.random.default_rng(SEED))
        if len(sel) == 0:
            continue
        amt = budget / len(sel)
        sh = np.zeros(n_tk)
        for j in sel:
            if not tradable[j]:
                continue
            q = np.floor(amt / px[j])
            need = q * px[j] * (1 + cost_pct / 100.0)
            if q <= 0 or need > cash:
                continue
            cash -= need
            sh[j] = q
        if sh.sum() > 0:
            tranches.append({"shares": sh, "die": reb_no + n_tranche})

    eq_s = pd.Series(equity, index=pd.to_datetime(dates)).dropna()
    if len(eq_s) != n_days - rebs[0]:
        raise RuntimeError("overlap: equity 시리즈 절단")
    return {"equity": eq_s}


# ── 성과 집계 ────────────────────────────────────────────────────────────

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


def diff_stats(d: pd.Series) -> tuple:
    """월별 차분의 평균·중위·t값·95% CI."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t = d.mean() / se if se and se > 0 else np.nan
    return d.mean(), d.median(), t, d.mean() - 1.96 * se, d.mean() + 1.96 * se


def run_market(market: str):
    cfg = MARKET_CFG[market]
    P = build_panel(market)
    bench = loading.load_bench(market)
    # 문서 §4: 백테스트 종료일 = min(가격 마지막 봉, 벤치 마지막 봉).
    # 벤치가 먼저 끝나면 ffill로 평탄 연장돼 마지막 달 비교가 불공정해진다.
    end = min(P["dates"][-1], bench["date"].max())
    keep = P["dates"] <= end
    for k in ("close", "close_ff", "open", "pret", "id", "dollar_vol", "bars"):
        P[k] = P[k][keep]
    P["dates"] = P["dates"][keep]
    bull_map = regime.bull_map(bench, sma=SMA_BENCH)
    bull = np.array([bull_map.get(d, False) for d in P["dates"]])
    bser = bench.set_index(pd.to_datetime(bench["date"]))["close"].astype(float)

    print(f"\n{'='*84}\n[{market.upper()}] 유니버스 {len(P['tickers']):,}종목 · "
          f"데이터 기준일 {P['dates'][-1]} · 유동성 상위 {cfg['universe_n']} · "
          f"후보 {POOL_N} → 선정 {HOLD_N} · 편도비용 {cfg['cost_pct']}%\n{'='*84}")

    out = {}
    for net in (True, False):
        for g in GROUPS:
            out[(g, net)] = simulate(P, bull, g, cfg, cfg["cost_pct"] if net else 0.0,
                                     np.random.default_rng(SEED))

    a_eq = out[("A", True)]["equity"]
    bench_eq = bser.reindex(a_eq.index).ffill().dropna()
    a_eq = a_eq.reindex(bench_eq.index).dropna()
    bench_eq = bench_eq.reindex(a_eq.index)

    rows = {}
    for g in GROUPS:
        eq = out[(g, True)]["equity"].reindex(a_eq.index).ffill()
        rows[g] = stats(eq, bench_eq)
        for k in ("turnover", "invested", "n_pos", "zombie", "ties"):
            rows[g][k] = out[(g, True)][k]
    rows["B"] = stats(bench_eq, bench_eq)
    rows["B"].update(turnover=0.0, invested=1.0, n_pos=np.nan, zombie=0, ties=0.0)

    print(f"{'그룹':<10} {'월수':>4} {'월초과평균':>9} {'월초과중위':>9} {'CAGR':>7} "
          f"{'변동성':>6} {'Sharpe':>6} {'MDD':>7} {'회전율':>7} {'투자비중':>7} {'종목수':>6}")
    print("-" * 96)
    label = {"A": "A ID오름", "B": "B 벤치마크", "C": "C PRET순",
             "D": "D 무작위", "E": "E ID내림"}
    for g in ["A", "B", "C", "D", "E"]:
        s = rows[g]
        print(f"{label[g]:<10} {s['n_months']:>4} {s['exc_mean']:>8.3f}% {s['exc_med']:>8.3f}% "
              f"{s['cagr']:>6.2f}% {s['vol']:>5.1f}% {s['sharpe']:>6.2f} "
              f"{s['mdd']:>6.1f}% {s['turnover']:>6.2f}x {s['invested']*100:>6.1f}% "
              f"{s['n_pos']:>5.1f}")

    def dif(g1, g2, net=True):
        e1 = out[(g1, net)]["equity"].reindex(a_eq.index).ffill()
        e2 = out[(g2, net)]["equity"].reindex(a_eq.index).ffill()
        return diff_stats((monthly(e1) - monthly(e2)).dropna() * 100)

    print(f"\n[핵심 차분 — 월별 수익률 차, %p · t값 · 95% CI]")
    for lbl, g1, g2, net in [("② A−C 비용후 ★", "A", "C", True),
                             ("③ A−D gross ★", "A", "D", False),
                             ("④ A−E 비용후 ★", "A", "E", True),
                             ("   A−D net", "A", "D", True)]:
        mu, md, t, lo, hi = dif(g1, g2, net)
        print(f"  {lbl:<16} 평균 {mu:>7.3f}%p · 중위 {md:>7.3f}%p · "
              f"t={t:>5.2f} · CI[{lo:>6.3f}, {hi:>6.3f}]")

    print(f"\n[사전 판정 기준 대조]")
    s = rows["A"]
    ok1 = s["exc_mean"] > 0 and s["exc_med"] > 0
    print(f"  ① A vs B 평균>0 & 중위>0 (비용후): 평균 {s['exc_mean']:+.3f}%p · "
          f"중위 {s['exc_med']:+.3f}%p → {'✓' if ok1 else '✗'}")
    for lbl, g1, g2, net in [("②", "A", "C", True), ("③", "A", "D", False),
                             ("④", "A", "E", True)]:
        mu, md, t, lo, hi = dif(g1, g2, net)
        print(f"  {lbl} A−{g2}{' gross' if not net else ''} 평균>0: {mu:+.3f}%p "
              f"{'✓' if mu > 0 else '✗'} (중위 {md:+.3f}%p, t={t:.2f})")

    # 부가 — D 시드 3개, 익일시가 민감도, 레짐 비율
    # 6개월 중첩 트랑슈 민감도 (§3 사전 등록, 판정 미사용)
    tr = simulate_overlap(P, bull, cfg, cfg["cost_pct"])
    st = stats(tr["equity"].reindex(a_eq.index).ffill(), bench_eq)
    print(f"\n[민감도] A 6개월 중첩 트랑슈(논문 세팅): 월초과 평균 {st['exc_mean']:+.3f}%p · "
          f"중위 {st['exc_med']:+.3f}%p · CAGR {st['cagr']:.2f}%")

    print(f"[부가] 20위 경계 동점 슬롯 월평균: A {rows['A']['ties']:.2f}개 / {HOLD_N}")
    # 선정 종목의 '사전' 실현 변동성 — 선정 시점 직전 252봉 일간수익률 표준편차를
    # 연율화(×√252)하고, 선정 20종목 평균 → 강세 리밸일 평균 순으로 집계한다.
    # §6 부수 발견(ID가 사실상 저변동성 필터)의 근거이므로 여기서 함께 산출한다.
    print("[부가] 선정 종목 사전 252일 실현 변동성(연율):", end=" ")
    c = P["close"]
    ret_all = np.full_like(c, np.nan)
    ret_all[1:] = c[1:] / c[:-1] - 1.0
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_FAR, 1)
    for g in GROUPS:
        rng_g, vals = np.random.default_rng(SEED), []
        for i in month_end_indices(P["dates"], start_i):
            if not bull[i]:
                continue
            sel = select(P, i, g, cfg, rng_g)
            if len(sel) == 0:
                continue
            w = ret_all[max(0, i - LB_FAR + 1):i + 1, sel]
            with np.errstate(all="ignore"):
                v = np.nanstd(w, axis=0, ddof=1) * np.sqrt(252)
            if np.isfinite(v).any():
                vals.append(np.nanmean(v))
        print(f"{g} {np.mean(vals)*100:.1f}%", end="  ")
    # ②③④ 차분의 음/영/양 달 수 — 중위가 0으로 축퇴하므로 부호 분포를 함께 본다
    print("\n[부가] 차분의 음/영/양 달 수:", end=" ")
    for lbl, g1, g2, net in [("A−C", "A", "C", True), ("A−D gross", "A", "D", False),
                             ("A−E", "A", "E", True)]:
        e1 = out[(g1, net)]["equity"].reindex(a_eq.index).ffill()
        e2 = out[(g2, net)]["equity"].reindex(a_eq.index).ffill()
        d = (monthly(e1) - monthly(e2)).dropna()
        print(f"{lbl} {int((d < 0).sum())}/{int((d == 0).sum())}/{int((d > 0).sum())}", end="  ")
    print()
    print(f"[부가] D 시드별 A−D gross 평균:", end=" ")
    for sd in SEEDS_D:
        r = simulate(P, bull, "D", cfg, 0.0, np.random.default_rng(sd))
        e = r["equity"].reindex(a_eq.index).ffill()
        ag = out[("A", False)]["equity"].reindex(a_eq.index).ffill()
        print(f"{sd}: {(monthly(ag) - monthly(e)).dropna().mean()*100:+.3f}%p", end="  ")
    nx = simulate(P, bull, "A", cfg, cfg["cost_pct"], np.random.default_rng(SEED),
                  exec_next_open=True)
    sn = stats(nx["equity"].reindex(a_eq.index).ffill(), bench_eq)
    print(f"\n[민감도] A 익일시가 체결: 월초과 평균 {sn['exc_mean']:+.3f}%p · "
          f"중위 {sn['exc_med']:+.3f}%p · CAGR {sn['cagr']:.2f}%")
    seg = np.array([bull_map.get(d, False) for d in P["dates"]
                    if a_eq.index[0].date() <= d <= a_eq.index[-1].date()])
    print(f"[레짐] 강세(지수>SMA200) 거래일 비율 {seg.mean()*100:.1f}% "
          f"({int(seg.sum()):,}/{len(seg):,}일)")
    print(f"[좀비] 리밸일에 거래 불가였던 보유 건수: A {rows['A']['zombie']}건")
    print(f"[구간] {a_eq.index[0].date()} ~ {a_eq.index[-1].date()} "
          f"(독립 표본 {rows['A']['n_months']}개월)")
    return rows


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest():
    # ① PRET·ID를 정답을 아는 계열로 확인
    n = 400
    up = 100 * np.cumprod(np.r_[1.0, np.where(np.arange(1, n) % 2 == 0, 1.02, 1.01)])
    df = pd.DataFrame({"T": up}, index=pd.bdate_range("2015-01-01", periods=n))
    P = _panel_from(df)
    t = n - 1
    want_pret = up[t - LB_NEAR] / up[t - LB_FAR] - 1
    assert abs(P["pret"][t, 0] - want_pret) < 1e-12, "PRET 불일치"
    # 전 구간 상승 → %neg=0, %pos=231/231 → ID = -1
    assert abs(P["id"][t, 0] - (-1.0)) < 1e-12, f"ID {P['id'][t,0]} != -1"

    # ② 하락일이 섞인 계열: 느린 루프와 대조
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.02, n)
    px = 100 * np.cumprod(1 + r)
    df2 = pd.DataFrame({"T": px}, index=pd.bdate_range("2015-01-01", periods=n))
    P2 = _panel_from(df2)
    for tt in (LB_FAR + 5, 350, n - 1):
        seg = px[tt - LB_FAR + 1:tt - LB_NEAR + 1] / px[tt - LB_FAR:tt - LB_NEAR] - 1
        assert len(seg) == N_RET, f"수익률 개수 {len(seg)} != {N_RET}"
        want = np.sign(px[tt - LB_NEAR] / px[tt - LB_FAR] - 1) * \
            ((seg < 0).sum() - (seg > 0).sum()) / N_RET
        assert abs(P2["id"][tt, 0] - want) < 1e-12, f"t={tt} ID {P2['id'][tt,0]} vs {want}"

    # ③ 유효일 하한: 결측을 많이 넣으면 ID가 NaN
    px3 = px.copy()
    df3 = pd.DataFrame({"T": px3}, index=df2.index)
    df3.iloc[100:130] = np.nan
    P3 = _panel_from(df3)
    assert np.isnan(P3["id"][320, 0]), "유효일 부족인데 ID가 계산됨"

    # ④ 동점 tie-break: ID 동일 시 티커 오름차순, 방향 반전(E)에도 티커는 오름차순
    _selftest_tiebreak()

    # ⑤ 시뮬레이터: 단일 종목·항상 강세 → 매수 후 보유와 등가
    _selftest_single()

    # ⑥ 다종목 — 느린 참조 구현과 최종 자본 대조
    _selftest_multi()
    print("selftest: 6개 항목 통과 (PRET·ID 정답/루프 대조·유효일 하한·"
          "동점 규칙·매수후보유 등가·다종목 참조 대조)")


def _panel_from(df: pd.DataFrame) -> dict:
    """DataFrame(date × ticker) → build_panel과 동일한 지표 계산 (테스트용)."""
    c = df.to_numpy(float)
    far, near = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[LB_FAR:] = c[:-LB_FAR]
    near[LB_NEAR:] = c[:-LB_NEAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        pret = near / far - 1.0
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    pos = _count_window(np.isfinite(ret) & (ret > 0), N_RET)
    neg = _count_window(np.isfinite(ret) & (ret < 0), N_RET)
    val = _count_window(np.isfinite(ret), N_RET)
    sh = lambda a: np.vstack([np.full((LB_NEAR, a.shape[1]), np.nan), a[:-LB_NEAR]])
    pos, neg, val = sh(pos), sh(neg), sh(val)
    with np.errstate(invalid="ignore"):
        idv = np.sign(pret) * ((neg - pos) / N_RET)
    idv = np.where(val >= MIN_VALID_RET, idv, np.nan)
    return {"dates": np.asarray(df.index.date), "tickers": np.asarray(df.columns, dtype=str),
            "close": c, "close_ff": df.ffill().to_numpy(float), "open": c,
            "pret": pret, "id": idv, "dollar_vol": c * 1e6,
            "bars": df.notna().cumsum().to_numpy(float)}


def _selftest_tiebreak():
    k = 6
    P = {"tickers": np.array(["E", "D", "C", "B", "A", "F"]),
         "close": np.ones((1, k)), "bars": np.full((1, k), 1e9),
         "pret": np.full((1, k), 0.5), "id": np.zeros((1, k)),      # 전원 동점
         "dollar_vol": np.ones((1, k))}
    P["pret"][0, 5] = 0.9                       # F만 PRET 높음
    got_a = select(P, 0, "A", {"universe_n": k}, np.random.default_rng(0))
    got_e = select(P, 0, "E", {"universe_n": k}, np.random.default_rng(0))
    # ID 전원 동점 → A·E 모두 티커 오름차순으로 앞에서부터
    assert list(P["tickers"][got_a][:3]) == ["A", "B", "C"], P["tickers"][got_a]
    assert list(P["tickers"][got_e][:3]) == ["A", "B", "C"], P["tickers"][got_e]
    # C는 PRET 내림차순 → F가 1순위
    got_c = select(P, 0, "C", {"universe_n": k}, np.random.default_rng(0))
    assert P["tickers"][got_c][0] == "F", P["tickers"][got_c]


def _fixture_single(px: np.ndarray) -> dict:
    df = pd.DataFrame({"T": px}, index=pd.bdate_range("2015-01-01", periods=len(px)))
    P = _panel_from(df)
    P["pret"] = np.full_like(P["pret"], 0.5)     # 항상 적격
    P["id"] = np.full_like(P["id"], -0.5)
    P["bars"] = np.full_like(P["bars"], 1e9)
    return P


def _selftest_single():
    """단일 종목으로 매매 수량·비중 규칙을 정확한 기대값과 대조한다.

    주의: 이 설계는 **매월 동일비중 재조정**이라, 오르는 종목 하나만 담으면 목표
    비중(1/20)을 넘는 만큼 매달 부분 매도된다. 따라서 매수 후 보유와 등가가 아니다.
    """
    m = 400
    # (a) 가격 불변 → 재조정할 것이 없으므로 자본 불변, 수량은 목표 금액 그대로
    flat = np.full(m, 100.0)
    P = _fixture_single(flat)
    r = simulate(P, np.ones(m, dtype=bool), "A", {"universe_n": 1}, 0.0,
                 np.random.default_rng(0))
    assert abs(r["equity"].iloc[-1] - START_EQUITY) < 1e-6, r["equity"].iloc[-1]
    # 종목당 목표는 자산의 1/20 = 5%. invested는 전 기간 평균이라 첫 리밸일(매수 전
    # 0%) 하루가 섞여 5%를 살짝 밑돈다.
    want = np.floor(START_EQUITY / HOLD_N / 100.0) * 100.0 / START_EQUITY   # = 0.05
    assert want * 0.99 < r["invested"] <= want + 1e-12, \
        f"목표 비중 {want:.4f} 대비 실제 평균 {r['invested']:.4f}"

    # (b) 상승 종목 → 부분 매도가 실제로 일어나 매수 후 보유보다 자본이 작아야 한다
    up = np.linspace(100.0, 300.0, m)
    P2 = _fixture_single(up)
    r2 = simulate(P2, np.ones(m, dtype=bool), "A", {"universe_n": 1}, 0.0,
                  np.random.default_rng(0))
    eq2 = r2["equity"]
    i0 = list(pd.to_datetime(P2["dates"])).index(eq2.index[0])
    hold_only = START_EQUITY + np.floor(START_EQUITY / HOLD_N / up[i0]) * (up[-1] - up[i0])
    assert eq2.iloc[-1] < hold_only, "재조정이 일어나지 않았다"
    assert eq2.iloc[-1] > START_EQUITY, "상승장인데 자본이 늘지 않았다"

    # (c) 레짐이 항상 약세면 매수가 없어 자본 불변
    r3 = simulate(P2, np.zeros(m, dtype=bool), "A", {"universe_n": 1}, 0.0,
                  np.random.default_rng(0))
    assert abs(r3["equity"].iloc[-1] - START_EQUITY) < 1e-9

    # (d) 비용을 넣으면 자본이 줄어든다
    r4 = simulate(P2, np.ones(m, dtype=bool), "A", {"universe_n": 1}, 0.5,
                  np.random.default_rng(0))
    assert r4["equity"].iloc[-1] < eq2.iloc[-1]


def _reference_sim(P, bull, cfg, cost_pct, rebs, seed):
    """simulate()의 A 그룹을 리스트·sorted()만으로 다시 구현한 참조본."""
    close, n_days, n_tk = P["close"], P["close"].shape[0], P["close"].shape[1]
    shares, cash, last = [0.0] * n_tk, START_EQUITY, START_EQUITY
    rebs = set(int(x) for x in rebs)
    for i in range(min(rebs), n_days):
        last = cash + sum(shares[j] * (P["close_ff"][i][j]
                                       if np.isfinite(P["close_ff"][i][j]) else 0.0)
                          for j in range(n_tk))
        if i not in rebs:
            continue
        px = close[i]
        tgt = [] if not bull[i] else list(select(P, i, "A", cfg,
                                                 np.random.default_rng(seed)))
        ts = set(int(x) for x in tgt)
        for j in range(n_tk):
            if shares[j] > 0 and int(j) not in ts and np.isfinite(px[j]):
                cash += shares[j] * px[j] * (1 - cost_pct / 100.0)
                shares[j] = 0.0
        if not tgt:
            continue
        eq = cash + sum(shares[j] * (P["close_ff"][i][j]
                                     if np.isfinite(P["close_ff"][i][j]) else 0.0)
                        for j in range(n_tk))
        amt = eq / HOLD_N
        for j in tgt:
            if shares[j] > 0 and np.isfinite(px[j]):
                cur = shares[j] * px[j]
                if cur - amt >= px[j]:
                    q = float(np.floor((cur - amt) / px[j]))
                    cash += q * px[j] * (1 - cost_pct / 100.0)
                    shares[j] -= q
        for j in tgt:
            if not np.isfinite(px[j]):
                continue
            cur = shares[j] * px[j]
            if amt - cur < px[j]:
                continue
            q = float(np.floor((amt - cur) / px[j]))
            need = q * px[j] * (1 + cost_pct / 100.0)
            if q > 0 and need <= cash:
                cash -= need
                shares[j] += q
    return last


def _selftest_multi():
    rng = np.random.default_rng(11)
    m, k = 420, 30
    dates = pd.bdate_range("2015-01-01", periods=m)
    px = 100 * np.cumprod(1 + rng.normal(0.0006, 0.02, (m, k)), axis=0)
    px[:60, 5] = np.nan                         # 늦게 상장
    df = pd.DataFrame(px, index=dates, columns=[f"T{j:02d}" for j in range(k)])
    P = _panel_from(df)
    bull = np.ones(m, dtype=bool)
    bull[260:300] = False
    cfg = {"universe_n": k}
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                  SMA_BENCH, LB_FAR, 1)
    rebs = month_end_indices(P["dates"], start_i)
    got = simulate(P, bull, "A", cfg, 0.1, np.random.default_rng(SEED))
    want = _reference_sim(P, bull, cfg, 0.1, set(rebs), SEED)
    assert abs(got["equity"].iloc[-1] - want) < 1e-6, \
        f"벡터화 {got['equity'].iloc[-1]:.6f} vs 참조 {want:.6f}"
    assert got["n_pos"] > 0, "매수가 한 건도 없으면 대조 의미가 없다"


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if arg == "--selftest":
        selftest()
    else:
        run_market(arg)
