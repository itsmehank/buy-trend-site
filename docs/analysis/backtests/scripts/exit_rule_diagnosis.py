"""H-013 — 추세 이탈 청산은 왜 성과를 깎았나 (진단).

설계 문서: docs/analysis/backtests/2026-08-02-exit-rule-diagnosis.md

H-009의 A 그룹을 **규칙 변경 없이** 재실행하되 매도·매수 이벤트를 계측해, SMA100
하회로 판 종목이 그 뒤에 어떻게 움직였는지를 랭킹 이탈 매도·대체 매수 바스켓과
비교한다.

**H-009 스크립트를 직접 고치지 않는 이유**
아카이브된 테스트의 재현물이 바뀌기 때문이다(PROTOCOL §5). 여기에 A 시뮬레이션을
복사해 계측만 추가하고, `--selftest`에서 **복사본이 H-009 A의 최종 자본을 그대로
재현하는지** 확인한다.

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/exit_rule_diagnosis.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/exit_rule_diagnosis.py kr
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/exit_rule_diagnosis.py us
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # docs/analysis
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # 저장소 루트

from btlib import loading, regime  # noqa: E402
# H-009의 A 규칙·파라미터를 그대로 가져온다 (규칙 변경 없음)
import clenow_momentum_ranking as h9  # noqa: E402

HORIZONS = (20, 60, 126)
REB_STEP = 5                      # 주간 리밸런싱 → NW lag 단위
GROUPS = ("SMA", "RANK", "GAP", "REPLACE", "HOLD")
#: 설계 §2.1.1의 REPLACE는 "SMA 단독 이벤트가 발생한 리밸일"의 신규 매수다.
#: 코드는 모든 리밸일의 신규 매수를 REPLACE로 기록하므로(③은 짝짓기가 자동으로
#: 걸러 결과가 같다), 기술 통계용으로 설계 정의 행을 따로 만든다.
REPLACE_ALIGNED = "REPLACE(SMA일)"


# ── 이벤트 계측 (H-009 A 시뮬레이션 복사 + 계측) ─────────────────────────

def simulate_with_events(P, bull, cfg, cost_pct):
    """H-009 `simulate(group='A')`와 동일 규칙. 매도 사유와 신규 매수를 함께 기록.

    반환: (최종 equity 시리즈, 이벤트 리스트)
      이벤트 = {"i": 리밸 인덱스, "j": 종목 인덱스, "kind": SMA|RANK|GAP|COMBO|REPLACE|HOLD}
    """
    close, dates = P["close"], P["dates"]
    n_days, n_tk = close.shape
    close_ff = P["close_ff"]
    start_i = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= h9.MIN_BARS)),
                  h9.SMA_BENCH, h9.REG_WIN, 1)
    rebs = h9.rebalance_dates(dates, start_i)
    reb_set = set(rebs)

    shares = np.zeros(n_tk)
    cash = h9.START_EQUITY
    equity = np.full(n_days, np.nan)
    events = []

    for i in range(rebs[0], n_days):
        val = float((shares * np.nan_to_num(close_ff[i])).sum())
        equity[i] = cash + val
        if i not in reb_set:
            continue
        eq = equity[i]
        exec_px = close[i]                      # 본안: 당일 종가 체결
        sc, atr_i = P["score"][i], P["atr"][i]
        sma_i, gap_i = P["sma"][i], P["gap"][i]
        valid = np.isfinite(exec_px) & (P["bars"][i] >= h9.MIN_BARS)

        dv_slice = P["dollar_vol"][max(0, i - h9.DV_WIN + 1):i + 1]
        with np.errstate(all="ignore"):
            dv_med = np.nanmedian(dv_slice, axis=0)
        univ = valid & np.isfinite(dv_med) & np.isfinite(sc)
        if univ.sum() == 0:
            continue
        n_keep = min(int(cfg["universe_n"]), int(univ.sum()))
        thr_dv = np.sort(dv_med[univ])[::-1][n_keep - 1]
        univ &= (dv_med >= thr_dv)

        top = np.zeros(n_tk, dtype=bool)
        u_idx = np.flatnonzero(univ)
        k = max(1, int(round(len(u_idx) * h9.TOP_PCT)))
        top[u_idx[np.argsort(sc[u_idx])[::-1][:k]]] = True
        eligible = univ & (exec_px > sma_i) & (gap_i == 0) & np.isfinite(atr_i) & (atr_i > 0)

        held = shares > 0
        # ── 청산 (H-009 A: 랭킹 이탈 | SMA100 하회 | 갭)
        sell = held & ~top
        sell |= held & (~np.isfinite(exec_px) | (exec_px <= sma_i) | (gap_i == 1))
        sell &= np.isfinite(exec_px)
        for j in np.flatnonzero(sell):
            reasons = []
            if not top[j]:
                reasons.append("RANK")
            if np.isfinite(exec_px[j]) and exec_px[j] <= sma_i[j]:
                reasons.append("SMA")
            if gap_i[j] == 1:
                reasons.append("GAP")
            kind = reasons[0] if len(reasons) == 1 else "COMBO"
            events.append({"i": i, "j": int(j), "kind": kind,
                           "reasons": tuple(reasons)})
        if sell.any():
            gross = (shares[sell] * exec_px[sell]).sum()
            cash += gross * (1 - cost_pct / 100.0)
            shares[sell] = 0.0
            held = shares > 0

        # 보유 지속 (HOLD) — 청산되지 않고 남은 종목
        for j in np.flatnonzero(held):
            events.append({"i": i, "j": int(j), "kind": "HOLD", "reasons": ()})

        if not bull[i]:
            continue
        # ── 신규 매수 (REPLACE)
        cand = np.flatnonzero(eligible & ~held)
        if len(cand) == 0:
            continue
        ranked = cand[np.argsort(sc[cand])[::-1]]
        for j in ranked[top[ranked]]:
            qty = np.floor(eq * h9.RISK_FACTOR / atr_i[j])
            if qty <= 0:
                continue
            need = qty * exec_px[j] * (1 + cost_pct / 100.0)
            if need > cash:
                break
            cash -= need
            shares[j] = qty
            events.append({"i": i, "j": int(j), "kind": "REPLACE", "reasons": ()})

    eq_s = pd.Series(equity, index=pd.to_datetime(dates)).dropna()
    if len(eq_s) != n_days - rebs[0]:
        raise RuntimeError(f"equity 절단: {len(eq_s)} != {n_days - rebs[0]}")
    return eq_s, events


# ── 사후 수익률 측정 ─────────────────────────────────────────────────────

def measure(P, bench_close, events):
    """문서 §2.2 — 초과수익_h = (종목 t+h/t − 1) − (벤치 t+h/t − 1).

    측정 단계에서만 종가 **또는** 시가가 0·음수인 봉을 NaN 처리한다(§2.2).
    t·t+20·t+60·t+126이 종목·벤치 **모두** 유효한 이벤트만 남긴다.
    """
    c = P["close"].copy()
    bad = (c <= 0) | (P["open"] <= 0)
    c = np.where(bad, np.nan, c)
    n_days = c.shape[0]
    b = bench_close

    rows = []
    for ev in events:
        i, j = ev["i"], ev["j"]
        if any(i + h >= n_days for h in HORIZONS):
            continue
        if not (np.isfinite(c[i, j]) and c[i, j] > 0 and np.isfinite(b[i]) and b[i] > 0):
            continue
        vals, ok = {}, True
        for h in HORIZONS:
            ct, bt = c[i + h, j], b[i + h]
            if not (np.isfinite(ct) and ct > 0 and np.isfinite(bt) and bt > 0):
                ok = False
                break
            vals[h] = (ct / c[i, j] - 1.0) - (bt / b[i] - 1.0)
        if ok:
            rows.append({"i": i, "j": j, "kind": ev["kind"], **vals})
    return pd.DataFrame(rows)


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """Bartlett 커널 NW 표준오차. 문서 §3의 수식 검증을 그대로 구현."""
    T = len(x)
    if T < 2:
        return np.nan
    d = x - x.mean()
    g0 = float((d * d).sum() / T)
    s = g0
    for j in range(1, min(lag, T - 1) + 1):
        gj = float((d[j:] * d[:-j]).sum() / T)
        s += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    s = max(s, 0.0)
    return float(np.sqrt(s / T))


def daily_series(df: pd.DataFrame, kind: str, h: int) -> pd.Series:
    """리밸일 단위 평균 계열 (그 그룹의 이벤트가 존재한 리밸일만)."""
    if kind == REPLACE_ALIGNED:      # 설계 §2.1.1: SMA 단독 이벤트가 있었던 리밸일만
        sma_days = set(df[df["kind"] == "SMA"]["i"])
        sub = df[(df["kind"] == "REPLACE") & (df["i"].isin(sma_days))]
    else:
        sub = df[df["kind"] == kind]
    return sub.groupby("i")[h].mean() if len(sub) else pd.Series(dtype=float)


def win_rate(df: pd.DataFrame, kind: str, h: int) -> float:
    """이벤트 단위 승률(초과수익>0 비율) — §3 부가 기록."""
    if kind == REPLACE_ALIGNED:
        sma_days = set(df[df["kind"] == "SMA"]["i"])
        sub = df[(df["kind"] == "REPLACE") & (df["i"].isin(sma_days))]
    else:
        sub = df[df["kind"] == kind]
    return float((sub[h] > 0).mean() * 100) if len(sub) else np.nan


def stat_block(s: pd.Series, h: int) -> dict:
    """일별 계열 → 평균·중위·NW CI. lag = ⌈h/5⌉."""
    if len(s) < 2:
        return {"n_days": len(s), "mean": np.nan, "med": np.nan,
                "lo": np.nan, "hi": np.nan}
    x = s.to_numpy(float)
    lag = int(np.ceil(h / REB_STEP))
    se = newey_west_se(x, lag)
    m = float(x.mean())
    return {"n_days": len(x), "mean": m, "med": float(np.median(x)),
            "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def paired_block(df: pd.DataFrame, a: str, b: str, h: int) -> dict:
    """②③ — 두 그룹이 모두 관측된 리밸일만 짝지어 차분 계열을 만든 뒤 NW."""
    sa, sb = daily_series(df, a, h), daily_series(df, b, h)
    common = sa.index.intersection(sb.index)
    if len(common) < 2:
        return {"n_days": len(common), "mean": np.nan, "lo": np.nan, "hi": np.nan,
                "dropped_a": len(sa) - len(common), "dropped_b": len(sb) - len(common)}
    d = (sa.loc[common] - sb.loc[common]).to_numpy(float)
    lag = int(np.ceil(h / REB_STEP))
    se = newey_west_se(d, lag)
    m = float(d.mean())
    return {"n_days": len(d), "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "dropped_a": len(sa) - len(common), "dropped_b": len(sb) - len(common)}


# ── 실행 ─────────────────────────────────────────────────────────────────

def run_market(market: str):
    cfg = h9.MARKET_CFG[market]
    P = h9.build_panel(market)
    bench = loading.load_bench(market)
    bmap = dict(zip(bench["date"], bench["close"].astype(float)))
    bser = np.array([bmap.get(d, np.nan) for d in P["dates"]])
    bull_map = regime.bull_map(bench, sma=h9.SMA_BENCH)
    bull = np.array([bull_map.get(d, False) for d in P["dates"]])

    eq, events = simulate_with_events(P, bull, cfg, cfg["cost_pct"])
    df = measure(P, bser, events)

    print(f"\n{'='*94}\n[{market.upper()}] 유니버스 {len(P['tickers']):,}종목 · "
          f"기준일 {P['dates'][-1]} · H-009 A 재실행(net·당일종가) · "
          f"편도비용 {cfg['cost_pct']}%\n{'='*94}")

    # E가 회피하는 매도의 건수 분해 (문서 §2.1)
    from collections import Counter
    cnt = Counter(tuple(sorted(e["reasons"])) for e in events if e["reasons"])
    avoid = {k: v for k, v in cnt.items() if "RANK" not in k}
    print(f"[건수] 전체 매도 {sum(cnt.values()):,} · "
          f"E가 회피하는 매도(~RANK ∧ (SMA∨GAP)) {sum(avoid.values()):,}")
    for k in sorted(avoid, key=lambda x: -avoid[x]):
        print(f"        {'∩'.join(k):<10} {avoid[k]:>7,}")
    print(f"[측정] t·t+20·t+60·t+126 모두 유효한 관측 {len(df):,} / 이벤트 {len(events):,}")
    for g in GROUPS:
        print(f"        {g:<8} {int((df['kind']==g).sum()):>8,}", end="")
    print()

    print(f"\n※ 아래 CI는 명목 95%지만 겹치는 창 + 작은 표본 때문에 실제 오류율이 더 높다"
          f" — '0 제외'를 유의 판정으로 읽지 말 것 (문서 §5 주의)")
    print(f"\n{'그룹':<15}{'h':>5}{'일수':>6}{'평균':>9}{'중위':>9}{'승률':>7}{'95% CI':>22}")
    print("-" * 76)
    res = {}
    for g in list(GROUPS) + [REPLACE_ALIGNED]:
        for h in HORIZONS:
            st = stat_block(daily_series(df, g, h), h)
            st["win"] = win_rate(df, g, h)
            res[(g, h)] = st
            if np.isfinite(st["mean"]):
                print(f"{g:<15}{h:>5}{st['n_days']:>6}{st['mean']*100:>8.2f}%"
                      f"{st['med']*100:>8.2f}%{st['win']:>6.1f}%"
                      f"   [{st['lo']*100:>6.2f}, {st['hi']*100:>6.2f}]")

    print(f"\n[핵심 차분 — 리밸일 짝지은 차분 계열, NW 보정]")
    pair = {}
    for lbl, a, b in (("② SMA−RANK", "SMA", "RANK"), ("③ SMA−REPLACE", "SMA", "REPLACE")):
        for h in HORIZONS:
            pb = paired_block(df, a, b, h)
            pair[(lbl, h)] = pb
            if np.isfinite(pb["mean"]):
                ci0 = pb["lo"] <= 0 <= pb["hi"]
                print(f"  {lbl:<14} h={h:>3}: {pb['mean']*100:>+7.2f}%p · "
                      f"CI [{pb['lo']*100:>+6.2f}, {pb['hi']*100:>+6.2f}] · 짝 {pb['n_days']:>3}일 "
                      f"{'(CI 0 포함)' if ci0 else '(CI 0 제외)'} · "
                      f"제외 {pb['dropped_a']}/{pb['dropped_b']}일")

    print(f"\n[사전 판정 기준]")
    ok_h = []
    for h in HORIZONS:
        s = res[("SMA", h)]
        c1 = np.isfinite(s["mean"]) and s["mean"] > 0 and s["med"] > 0 and not (s["lo"] <= 0 <= s["hi"])
        p2, p3 = pair[("② SMA−RANK", h)], pair[("③ SMA−REPLACE", h)]
        c2 = np.isfinite(p2["mean"]) and p2["mean"] > 0 and not (p2["lo"] <= 0 <= p2["hi"])
        c3 = np.isfinite(p3["mean"]) and p3["mean"] > 0 and not (p3["lo"] <= 0 <= p3["hi"])
        mark = "✓" if (c1 and c2 and c3) else "✗"
        if c1 and c2 and c3:
            ok_h.append(h)
        print(f"  h={h:>3}: ①{'✓' if c1 else '✗'} ②{'✓' if c2 else '✗'} "
              f"③{'✓' if c3 else '✗'} → {mark}")
    verdict = len(ok_h) >= 2 and 126 in ok_h
    print(f"  → 충족 수평선 {ok_h or '없음'} · "
          f"(2개 이상 ∧ h=126 포함) = **{'지지' if verdict else '미충족'}**")
    if ok_h and 126 not in ok_h:
        print("  ※ h=126이 빠졌으므로 결론은 '단기 반전과 구분 불가'로 기록한다(§3)")

    # 독립 표본수 — **백테스트 구간** 기준 (패널 전체 구간을 쓰면 과대평가된다)
    span = (eq.index[-1] - eq.index[0]).days / 365.25 * 252
    print(f"\n[부가] 독립 표본수 ≈ 백테스트 구간 {span:.0f}거래일 ÷ h → "
          + " · ".join(f"h={h}: {span/h:.0f}" for h in HORIZONS))
    # ①과 ③이 서로 다른 조건집합을 보는지 확인 (짝집합에서의 SMA)
    for h in (126,):
        sa, sr = daily_series(df, "SMA", h), daily_series(df, "REPLACE", h)
        common = sa.index.intersection(sr.index)
        if len(common) >= 2:
            print(f"[부가] h={h} SMA 전체 {len(sa)}일 평균 {sa.mean()*100:+.2f}% vs "
                  f"③ 짝집합 {len(common)}일 평균 {sa.loc[common].mean()*100:+.2f}% "
                  f"(제외 {len(sa)-len(common)}일 평균 "
                  f"{sa.drop(common).mean()*100:+.2f}%) — ①과 ③은 다른 조건집합이다")
    print(f"[구간] {eq.index[0].date()} ~ {eq.index[-1].date()}")
    return res, pair, df, events


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest():
    # ① NW SE — 문서 §3 수식 검증표를 코드로 재확인
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    plain = x.std(ddof=0) / np.sqrt(len(x))
    assert abs(newey_west_se(x, 0) - plain) < 1e-12, "lag=0이면 단순 SE"
    # 자기상관 없는 계열에서 lag를 줘도 단순 SE와 크게 다르지 않다
    assert abs(newey_west_se(x, 4) - plain) / plain < 0.35
    # 완전 양의 자기상관(상수 계열 + 미세 잡음)에서 SE가 커진다
    y = np.repeat(rng.normal(0, 1, 20), 25)
    assert newey_west_se(y, 26) > newey_west_se(y, 0) * 2, "겹치는 창이면 CI가 넓어져야"

    # ② 초과수익 정의 — 종목과 벤치가 같이 움직이면 0
    P = {"close": np.array([[100.0], [110.0], [121.0]]),
         "open": np.array([[100.0], [110.0], [121.0]]),
         "dates": np.arange(3)}
    b = np.array([50.0, 55.0, 60.5])
    global HORIZONS
    old = HORIZONS
    try:
        HORIZONS = (1, 2)
        got = measure(P, b, [{"i": 0, "j": 0, "kind": "SMA", "reasons": ("SMA",)}])
        assert abs(got.iloc[0][1]) < 1e-12 and abs(got.iloc[0][2]) < 1e-12, got
        # 종목 +10%, 벤치 +4% → +6%p
        P2 = {"close": np.array([[100.0], [110.0], [110.0]]),
              "open": np.array([[100.0], [110.0], [110.0]]), "dates": np.arange(3)}
        b2 = np.array([50.0, 52.0, 52.0])
        g2 = measure(P2, b2, [{"i": 0, "j": 0, "kind": "SMA", "reasons": ("SMA",)}])
        assert abs(g2.iloc[0][1] - 0.06) < 1e-12, g2.iloc[0][1]
        # t+h에 벤치가 없으면 관측 탈락
        b3 = np.array([50.0, np.nan, 52.0])
        assert len(measure(P2, b3, [{"i": 0, "j": 0, "kind": "SMA", "reasons": ("SMA",)}])) == 0
    finally:
        HORIZONS = old

    # ③ 짝짓기 — 두 그룹이 모두 있는 리밸일만
    df = pd.DataFrame([{"i": 1, "kind": "SMA", 20: 0.10}, {"i": 1, "kind": "RANK", 20: 0.04},
                       {"i": 2, "kind": "SMA", 20: 0.20},
                       {"i": 3, "kind": "SMA", 20: 0.02}, {"i": 3, "kind": "RANK", 20: 0.01}])
    pb = paired_block(df, "SMA", "RANK", 20)
    assert pb["n_days"] == 2 and abs(pb["mean"] - 0.035) < 1e-12, pb
    assert pb["dropped_a"] == 1 and pb["dropped_b"] == 0, pb

    # ④ 사유 코딩 — 정답을 아는 소형 패널로 단독/COMBO 분기 확인
    _selftest_reasons()

    print("selftest: 4개 항목 통과 (NW 극단값·초과수익 정의/결측·짝짓기·사유 코딩)")
    print("  ※ H-009 A 재현 검증은 `--verify <market>`로 실행한다(캐시 필요)")


def _selftest_reasons():
    """소형 패널로 SMA단독 / GAP단독 / 조합(COMBO)을 각각 만들어 확인.

    유니버스를 15종목으로 두는 이유: 상위 20% 규칙에서 `k = round(15×0.2) = 3`이라
    앞 3종목이 모두 매수 대상이 된다(3종목만 두면 k=1이라 1개만 사진다).
    """
    n, k = h9.MIN_BARS + 400, 15
    dates = pd.bdate_range("2015-01-01", periods=n).date
    close = np.tile(np.linspace(100.0, 200.0, n)[:, None], (1, k))
    score = np.tile(np.r_[np.array([3.0, 2.9, 2.8]), np.full(k - 3, 0.1)], (n, 1))
    P = {"dates": np.asarray(dates),
         "tickers": np.array([chr(ord("A") + i) for i in range(k)]),
         "close": close, "close_ff": close, "open": close,
         "score": score,
         "atr": np.full((n, k), 1.0),
         "sma": np.full((n, k), 0.0),          # 기본: 전 종목 SMA 위
         "gap": np.zeros((n, k)),
         "dollar_vol": np.full((n, k), 1e9),
         "bars": np.tile(np.arange(1, n + 1)[:, None], (1, k)).astype(float)}
    bull = np.ones(n, dtype=bool)
    # 먼저 3종목을 모두 사게 한 뒤, 다음 리밸에서 사유를 하나씩 부여한다
    rebs = h9.rebalance_dates(P["dates"],
                              max(h9.MIN_BARS, h9.SMA_BENCH, h9.REG_WIN, 1))
    r0, r1 = rebs[0], rebs[1]
    P["sma"][r1:, 0] = 1e9        # A: 종가 ≤ SMA100 → SMA
    P["gap"][r1:, 1] = 1.0        # B: 갭 → GAP
    P["score"][r1:, 2] = -1e9     # C: 점수 최하 → 랭킹 이탈 가능
    _, events = simulate_with_events(P, bull, {"universe_n": k}, 0.0)
    at_r1 = {e["j"]: e for e in events if e["i"] == r1 and e["kind"] != "REPLACE"}
    assert at_r1[0]["reasons"] == ("SMA",), at_r1[0]
    assert at_r1[1]["reasons"] == ("GAP",), at_r1[1]
    assert at_r1[0]["kind"] == "SMA" and at_r1[1]["kind"] == "GAP"
    # 사유가 2개 이상이면 COMBO
    P2 = {k2: (v.copy() if isinstance(v, np.ndarray) else v) for k2, v in P.items()}
    P2["sma"][r1:, 1] = 1e9       # B에 SMA도 부여 → GAP∩SMA
    _, ev2 = simulate_with_events(P2, bull, {"universe_n": k}, 0.0)
    b2 = [e for e in ev2 if e["i"] == r1 and e["j"] == 1][0]
    assert b2["kind"] == "COMBO" and set(b2["reasons"]) == {"GAP", "SMA"}, b2


def ci_error_rate(n_rep: int = 2000, seed: int = 20260802):
    """이 CI 절차의 **귀무 하 오류율**을 측정한다 (문서 §5 서두의 근거).

    겹치는 창 구조를 그대로 재현한다: 매 리밸일(5거래일 간격)마다 이후 h거래일의
    일간수익률 합을 관측하되, 일간수익률은 평균 0의 순수 잡음이다. 따라서 참값은 0이고
    명목 95% CI가 0을 제외하는 비율은 5%여야 한다.
    """
    rng = np.random.default_rng(seed)
    print(f"[CI 귀무 오류율] 반복 {n_rep:,}회 · 참값 0 · 명목 5%가 나와야 정상")
    print(f"{'h':>5}{'T(일수)':>9}{'lag':>5}{'0 제외 비율':>12}")
    for h, T in ((20, 148), (20, 96), (60, 148), (60, 70), (126, 182), (126, 148),
                 (126, 70), (126, 471)):
        lag = int(np.ceil(h / REB_STEP))
        hit = 0
        for _ in range(n_rep):
            # 리밸일 T개, 각 관측은 이후 h일 잡음의 합 → 창이 (h/5)일치만큼 겹친다
            daily = rng.normal(0.0, 1.0, T * REB_STEP + h)
            x = np.array([daily[i * REB_STEP:i * REB_STEP + h].sum() for i in range(T)])
            se = newey_west_se(x, lag)
            m = x.mean()
            if not (m - 1.96 * se <= 0 <= m + 1.96 * se):
                hit += 1
        print(f"{h:>5}{T:>9}{lag:>5}{hit / n_rep * 100:>11.1f}%")


def verify_reproduction(market: str):
    """복사본이 H-009 A의 최종 자본을 그대로 재현하는지 확인 (문서 §2.1)."""
    cfg = h9.MARKET_CFG[market]
    P = h9.build_panel(market)
    bench = loading.load_bench(market)
    bull_map = regime.bull_map(bench, sma=h9.SMA_BENCH)
    bull = np.array([bull_map.get(d, False) for d in P["dates"]])
    mine, events = simulate_with_events(P, bull, cfg, cfg["cost_pct"])
    orig = h9.simulate(P, bull, "A", cfg, cfg["cost_pct"], np.random.default_rng(h9.SEED))
    a, b = mine.iloc[-1], orig["equity"].iloc[-1]
    assert abs(a - b) < 1e-6, f"재현 실패: 복사본 {a:.6f} vs H-009 {b:.6f}"
    # 이벤트 정합성 — 최종 자본만 같아서는 (i, j) 기록이 옳은지 알 수 없다
    sells = [e for e in events if e["reasons"]]
    holds = [e for e in events if e["kind"] == "HOLD"]
    reps = [e for e in events if e["kind"] == "REPLACE"]
    assert all(len(e["reasons"]) >= 1 for e in sells)
    assert all(e["kind"] == "COMBO" for e in sells if len(e["reasons"]) > 1)
    assert all(e["kind"] == e["reasons"][0] for e in sells if len(e["reasons"]) == 1)
    # 매도된 종목은 그 리밸일에 HOLD로도 기록되면 안 된다
    sell_keys = {(e["i"], e["j"]) for e in sells}
    assert not (sell_keys & {(e["i"], e["j"]) for e in holds}), "매도와 HOLD 중복"
    print(f"[{market.upper()}] H-009 A 재현 확인: 최종 자본 {a:,.2f} (차이 {abs(a-b):.2e})")
    print(f"  이벤트 정합: 매도 {len(sells):,} · HOLD {len(holds):,} · "
          f"REPLACE {len(reps):,} · 사유/종류 일치 · 매도↔HOLD 중복 없음")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if arg == "--selftest":
        selftest()
    elif arg == "--verify":
        verify_reproduction(sys.argv[2])
    elif arg == "--ci-sim":
        ci_error_rate()
    else:
        run_market(arg)
