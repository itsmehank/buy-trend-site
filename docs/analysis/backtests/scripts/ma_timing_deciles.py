"""H-025 — Han·Yang·Zhou (2013) MA 타이밍 × 변동성 십분위. **원문 형태 그대로.**

원문(JFQA 48(5), pp.1433–1461)의 설계:
  · 변동성 십분위 포트폴리오를 **연 1회**(직전 연도 말) 구성. 동일가중.
  · 각 십분위의 **일별 가격지수** P_j(t) 와 그 **L일 이동평균** A_j(t).
  · 규칙: P_j(t−1) > A_j(t−1) 이면 t일 보유, 아니면 현금(원문은 30일 T-bill).
  · MAP_j = 타이밍 수익 − 매수후보유 수익.

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/ma_timing_deciles.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/ma_timing_deciles.py --run
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading

_N = NormalDist()

VOL_WIN = 250                 # 직전 1년 일간수익률 표준편차
MIN_BARS = 251
DV_WIN = 60
N_DECILE = 10
L_MAIN = 10
L_ROBUST = (20, 50, 100, 200)
UNIV = {"kr": 400, "us": 500}
COST = {"kr": 0.0014, "us": 0.0005}       # 편도
RF = 0.0                                   # §2.4-2


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)
    bad = (close <= 0) | (piv["open"] <= 0)          # PROTOCOL §1
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]
    c = close.to_numpy(float)
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    vol = pd.DataFrame(ret).rolling(VOL_WIN).std().to_numpy(float)
    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close": c, "ret": ret, "vol": vol,
            "dollar_vol": (close * piv["volume"]).to_numpy(float),
            "bars": close.notna().cumsum().to_numpy(float)}


def year_end_indices(dates: np.ndarray) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby(s.index.year).last().values]


def form_deciles(P, i: int, market: str) -> list[np.ndarray]:
    """직전 연도 말 i에서 변동성 십분위를 구성한다. 이후 1년간 고정."""
    px, vol = P["close"][i], P["vol"][i]
    ok = (np.isfinite(px) & (px > 0) & (P["bars"][i] >= MIN_BARS)
          & np.isfinite(vol) & (vol > 0))                 # NaN 변동성은 제외(§2.2.1)
    if ok.sum() < N_DECILE * 5:
        return []
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dvm = np.nanmedian(dv, axis=0)
    ok &= np.isfinite(dvm)
    cand = np.flatnonzero(ok)
    keep = min(UNIV[market], len(cand))
    thr = np.sort(dvm[cand])[::-1][keep - 1]
    cand = cand[dvm[cand] >= thr]
    tk = P["tickers"]
    order = np.array(sorted(cand, key=lambda j: (vol[j], tk[j])), dtype=int)
    return [np.asarray(g, dtype=int) for g in np.array_split(order, N_DECILE)]


def decile_index(P, members: np.ndarray, a: int, b: int) -> np.ndarray:
    """[a+1, b] 구간의 일별 동일가중 수익률 (§2.1 산식)."""
    seg = P["ret"][a + 1:b + 1][:, members]
    out = np.zeros(seg.shape[0])
    for t in range(seg.shape[0]):
        r = seg[t][np.isfinite(seg[t])]
        out[t] = float(r.mean()) if len(r) else 0.0
    return out


def ma_timing(rets: np.ndarray, L: int, cost: float, rf: float | None = None):
    """일별 수익률 계열에 MA 타이밍을 적용한다.

    P(t)를 누적곱으로 만들고, P(t−1) > A(t−1) 이면 t일 보유.
    반환: (타이밍 일별수익, 매수후보유 일별수익, 전환 횟수, 보유일 비율)
    """
    rf = RF if rf is None else rf
    n = len(rets)
    P = np.empty(n + 1)
    P[0] = 100.0
    P[1:] = 100.0 * np.cumprod(1.0 + rets)
    A = pd.Series(P).rolling(L).mean().to_numpy()
    hold = np.zeros(n, dtype=bool)
    for t in range(n):
        k = t                      # P[k] = t−1 시점 가격 (P[0]이 시작)
        if k >= L - 1 and np.isfinite(A[k]):
            hold[t] = P[k] > A[k]
    tim = np.where(hold, rets, rf)
    # 전환 비용: 상태가 바뀐 날. 첫날은 현금→(보유면) 진입
    prev = np.concatenate([[False], hold[:-1]])
    switch = hold != prev
    tim = tim - switch * cost
    bh = rets.copy()
    bh[0] -= cost                  # 매수후보유도 첫날 진입 비용(§2.2.1)
    return tim, bh, int(switch.sum()), float(hold.mean())


def run_market(market: str, L: int = L_MAIN, cost: float | None = None,
               rf: float | None = None):
    P = build_panel(market)
    ye = year_end_indices(P["dates"])
    ye = [i for i in ye if P["bars"][i].max() >= MIN_BARS]
    # 십분위 **슬롯**별로 연도 구간의 일별 수익률을 이어붙여 연속 계열을 만든다.
    # (문서 §2.1: "재정렬 경계에서 P_j 는 연속이며, 구성만 교체된다")
    segs = {j: [] for j in range(N_DECILE)}
    dseg = []
    for a, b in zip(ye[:-1], ye[1:]):
        decs = form_deciles(P, a, market)
        if not decs or len(decs) != N_DECILE:
            continue
        n = b - a
        if n < 5:
            continue
        for j, mem in enumerate(decs):
            segs[j].append(decile_index(P, mem, a, b) if len(mem)
                           else np.zeros(n))
        dseg.append(P["dates"][a + 1:b + 1])
    if not dseg:
        return {"_hold": np.nan}
    dts = pd.to_datetime(np.concatenate(dseg))
    res, holds, sws = {}, [], []
    for j in range(N_DECILE):
        r = np.concatenate(segs[j])
        tim, bh, sw, hd = ma_timing(
            r, L, COST[market] if cost is None else cost, rf)
        holds.append(hd)
        sws.append(sw)
        d = pd.Series(tim - bh, index=dts)
        m = d.resample("ME").sum() * 100          # 월별 MAP (%p)
        res[j] = {"map_m": m.to_numpy(), "ann": float(np.mean(tim - bh)) * 252 * 100,
                  "n": len(m)}
    res["_hold"] = float(np.mean(holds))
    res["_switch"] = float(np.mean(sws))
    res["_years"] = len(dts) / 252
    return res


def cmd_run():
    K = 2.240890                                   # Bonferroni α=0.05/2 양측
    print("=" * 96)
    print("[H-025] MA 타이밍 × 변동성 십분위 — Han·Yang·Zhou (2013) 원문 형태")
    print(f"  L={L_MAIN} · 연 1회 재정렬 · 동일가중 · rf=0 · 비용 차감")
    print(f"  판정: 통합 표본 동일가중 MAP · Bonferroni |t| > {K:.3f}")
    print("=" * 96)
    pooled = []
    for m in ("kr", "us"):
        r = run_market(m)
        ds = [r[j] for j in range(N_DECILE) if j in r]
        print(f"\n── {m.upper()} ──  보유일 비율 {r['_hold']*100:.1f}%"
              f" · 전환 {r['_switch']:.0f}회 ({r['_switch']/r['_years']:.1f}회/년)"
              f" · {r['_years']:.1f}년")
        print(f"{'십분위':>7}{'월 MAP':>10}{'연환산':>10}{'t':>8}{'개월':>7}")
        for j in range(N_DECILE):
            if j not in r:
                continue
            x = r[j]["map_m"]
            t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            print(f"{j+1:>7}{x.mean():>+9.3f}%{r[j]['ann']:>+9.2f}%{t:>8.2f}{len(x):>7}")
        L_ = min(len(d["map_m"]) for d in ds)
        eq = np.mean([d["map_m"][:L_] for d in ds], axis=0)
        pooled.append(eq)
        t = eq.mean() / (eq.std(ddof=1) / np.sqrt(len(eq)))
        rho = np.corrcoef(np.arange(len(ds)),
                          [d["map_m"][:L_].mean() for d in ds])[0, 1]
        print(f"  동일가중 MAP  월 {eq.mean():+.3f}%p · 연 {eq.mean()*12:+.2f}%p · t={t:.2f}")
        print(f"  십분위 순위상관 ρ = {rho:+.3f}")
    p = np.concatenate(pooled)
    tp = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    print(f"\n{'='*96}")
    print(f"[통합 표본 판정] n={len(p)}개월 · 월 {p.mean():+.3f}%p · 연 {p.mean()*12:+.2f}%p"
          f" · **t = {tp:.2f}**")
    print(f"  조건① |t| > {K:.3f} → {'**충족**' if abs(tp) > K else '미충족(측정 불가)'}"
          + ("  · 부호 음수 → **기각(원문 반대)**" if tp < -K else ""))


def selftest():
    # ① MA 정의 — 원문 식 (P_{t−(L−1)}+…+P_t)/L
    P = np.arange(1.0, 21.0)
    A = pd.Series(P).rolling(10).mean().to_numpy()
    assert abs(A[9] - P[0:10].mean()) < 1e-12
    assert abs(A[19] - P[10:20].mean()) < 1e-12

    # ② look-ahead 없음 — t 이후를 지워도 t일 결정이 같다
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01,
                  0.02, -0.02, 0.01, 0.03, -0.01])
    t_full, _, _, _ = ma_timing(r, 5, 0.0)
    for cut in (11, 13):
        t_cut, _, _, _ = ma_timing(r[:cut], 5, 0.0)
        assert np.allclose(t_full[:cut], t_cut), cut

    # ③ 단조 증가 → 신호 항상 참 → MAP = 0 (gross)
    up = np.full(60, 0.01)
    tim, bh, sw, hd = ma_timing(up, 10, 0.0)
    # MA 워밍업(L−1일) 동안은 신호가 없어 타이밍 팔이 현금이다. 그 이후로 한정한다.
    assert abs(float(np.sum(tim[10:] - bh[10:]))) < 1e-12, float(np.sum(tim[10:] - bh[10:]))
    assert bool(np.all(tim[10:] == bh[10:]))
    assert hd > 0.8

    # ④ 단조 감소 → 신호 **항상 거짓** → 계속 현금 → MAP > 0
    dn = np.full(60, -0.01)
    tim2, bh2, _, hd2 = ma_timing(dn, 10, 0.0)
    assert hd2 == 0.0, hd2                       # "대부분"이 아니라 항상 거짓
    assert float(np.sum(tim2 - bh2)) > 0.5

    # ⑤ P_j(t) 산식 — 2종목 소형 예제
    Pp = {"ret": np.array([[np.nan, np.nan], [0.10, 0.00], [0.00, 0.20], [-0.10, 0.10]])}
    got = decile_index(Pp, np.array([0, 1]), 0, 3)
    assert np.allclose(got, [0.05, 0.10, 0.00]), got
    Pn = {"ret": np.array([[np.nan, np.nan], [0.10, np.nan]])}   # 한 종목 정지
    assert np.allclose(decile_index(Pn, np.array([0, 1]), 0, 1), [0.10])

    # ⑥ 비용 — 전환 없으면 0, 전환 1회당 편도 1회
    flat = np.full(60, 0.01)
    t0, _, sw0, _ = ma_timing(flat, 10, 0.0)
    t1, _, sw1, _ = ma_timing(flat, 10, 0.002)
    assert sw0 == sw1 == 1                        # 첫 진입 1회뿐
    assert abs((t0.sum() - t1.sum()) - 0.002) < 1e-12

    # ⑦ 십분위 분할 — 403종목
    sizes = [len(g) for g in np.array_split(np.arange(403), 10)]
    assert sizes == [41, 41, 41, 40, 40, 40, 40, 40, 40, 40], sizes
    assert sum(sizes) == 403

    # ⑧ 동점·NaN — 동점은 티커 순, NaN 변동성은 제외
    Pv = {"close": np.array([[10.0] * 4]), "vol": np.array([[0.2, 0.2, np.nan, 0.1]]),
          "bars": np.array([[999.0] * 4]), "tickers": np.array(["b", "a", "c", "d"]),
          "dollar_vol": np.array([[100.0] * 4])}
    ok = np.isfinite(Pv["vol"][0]) & (Pv["vol"][0] > 0)
    assert not ok[2]                              # NaN 제외
    cand = np.flatnonzero(ok)
    order = sorted(cand, key=lambda j: (Pv["vol"][0][j], Pv["tickers"][j]))
    assert order == [3, 1, 0], order              # d(0.1) → a(0.2) → b(0.2)

    print("selftest: 8개 항목 통과 (MA정의·look-ahead·단조증가MAP0·단조감소항상거짓·"
          "지수산식·비용·십분위분할·동점NaN)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--run":
        cmd_run()
    else:
        print(__doc__)
