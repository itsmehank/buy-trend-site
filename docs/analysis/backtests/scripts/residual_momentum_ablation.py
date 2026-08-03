"""H-015 잔차 모멘텀 — 겹치는 대조군 ablation.

PROTOCOL §3.1이 요구하는 **사전 MDE 측정**을 위한 `--power` 모드를 갖는다.

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/residual_momentum_ablation.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/residual_momentum_ablation.py --power

**--power 는 SE·MDE·겹침만 출력한다. 평균·t값·부호를 계산조차 하지 않는다.**
사전등록 오염을 막기 위한 의도적 제약이다(문서 §3.0). 차분 시계열의 평균을
구하는 코드 경로가 이 모드에 존재하지 않는다.
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading, regime, costs

_N = NormalDist()
K_MDE = _N.inv_cdf(0.975) + _N.inv_cdf(0.80)      # 2.801585 (α=5%, 검정력 80%)
E_STAR, E_STAR2 = 0.33, 0.17                       # 월 %p — H-014 §2.3

# ── 문서 §2 고정 파라미터 ────────────────────────────────────────────────
LB_FAR, LB_NEAR = 252, 21
W_LEN = LB_FAR - LB_NEAR                  # 231 — 잔차 창 [t-251, t-21]
REG_LEN = 750                             # 회귀 구간 [t-750, t-1] (36개월)
REG_LEN_24 = 500                          # 24개월 민감도
MIN_VALID_REG_FRAC = 700 / 750            # 유효 수익률 하한 비율 (§2.2)
MIN_VALID_W = 220
DV_WIN = 60
POOL_N = 90
HOLD_N = 20
SMA_BENCH = 200
START_EQUITY = 1e8

MARKET_CFG = {
    "us": {"universe_n": 500, "cost_pct": costs.COST_PCT["US"] / 2},
    "kr": {"universe_n": 300, "cost_pct": costs.COST_PCT["KR"] / 2},
}


# ── 패널 ─────────────────────────────────────────────────────────────────
def _csum(a: np.ndarray) -> np.ndarray:
    """앞에 0행을 붙인 누적합 — 구간합을 c[b]-c[a] 로 얻는다."""
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def build_panel(market: str, reg_len: int = REG_LEN) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # KR 거래정지 봉: open=high=low=0, close만 이월. OR 마스킹 (PROTOCOL §1)
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    n_days, n_tk = c.shape

    # PRET = close[t-21]/close[t-252] - 1
    far, near = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[LB_FAR:], near[LB_NEAR:] = c[:-LB_FAR], c[:-LB_NEAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        pret = near / far - 1.0

    # 일간수익률
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0

    # 시장수익률 — 벤치 종가를 가격 축에 재색인 후 ffill, 결측일은 관측에서 제외
    bdf = loading.load_bench(market)
    bench = (pd.Series(bdf["close"].to_numpy(float),
                       index=pd.to_datetime(bdf["date"]))
             .reindex(pd.to_datetime(idx)).ffill())
    bm = bench.to_numpy(float)
    rm = np.full(n_days, np.nan)
    rm[1:] = bm[1:] / bm[:-1] - 1.0

    ok = np.isfinite(ret) & np.isfinite(rm)[:, None]      # 공통 유효 관측
    R = np.where(ok, ret, 0.0)
    M = np.where(ok, np.broadcast_to(rm[:, None], c.shape), 0.0)
    one = ok.astype(float)

    C = {k: _csum(v) for k, v in
         {"n": one, "y": R, "x": M, "xy": R * M, "xx": M * M, "yy": R * R}.items()}

    def seg(k, lo, hi):
        """구간합 [lo, hi) — 행 인덱스 기준. 범위 밖은 NaN."""
        return C[k][hi] - C[k][lo]

    resmom = np.full((n_days, n_tk), np.nan)
    beta_out = np.full((n_days, n_tk), np.nan)
    min_reg = MIN_VALID_REG_FRAC * reg_len

    # 판단일 t에서 회귀 R=[t-reg_len, t-1], 잔차 창 W=[t-251, t-21]
    for t in range(reg_len + 1, n_days):
        lo_r, hi_r = t - reg_len, t                 # [t-reg_len, t-1] → 슬라이스 끝 t
        n_r = seg("n", lo_r, hi_r)
        sy, sx = seg("y", lo_r, hi_r), seg("x", lo_r, hi_r)
        sxy, sxx = seg("xy", lo_r, hi_r), seg("xx", lo_r, hi_r)
        with np.errstate(all="ignore"):
            cov = sxy / n_r - (sx / n_r) * (sy / n_r)
            var = sxx / n_r - (sx / n_r) ** 2
            beta = np.where(var > 0, cov / var, np.nan)
            alpha = sy / n_r - beta * (sx / n_r)

        lo_w, hi_w = t - (LB_FAR - 1), t - LB_NEAR + 1    # [t-251, t-21]
        n_w = seg("n", lo_w, hi_w)
        wy, wx = seg("y", lo_w, hi_w), seg("x", lo_w, hi_w)
        wxy, wxx, wyy = (seg("xy", lo_w, hi_w), seg("xx", lo_w, hi_w),
                         seg("yy", lo_w, hi_w))
        with np.errstate(all="ignore"):
            s_eps = wy - n_w * alpha - beta * wx
            # Σε² = Σy² - 2αΣy - 2βΣxy + nα² + 2αβΣx + β²Σx²
            s_eps2 = (wyy - 2 * alpha * wy - 2 * beta * wxy + n_w * alpha ** 2
                      + 2 * alpha * beta * wx + beta ** 2 * wxx)
            var_eps = (s_eps2 - s_eps ** 2 / n_w) / (n_w - 1)
            sd = np.sqrt(np.where(var_eps > 0, var_eps, np.nan))
            score = s_eps / sd
        good = (n_r >= min_reg) & (n_w >= MIN_VALID_W) & np.isfinite(score)
        resmom[t] = np.where(good, score, np.nan)
        beta_out[t] = np.where(good, beta, np.nan)

    return {
        "dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
        "close": c, "close_ff": close.ffill().to_numpy(float),
        "open": piv["open"].to_numpy(float),
        "pret": pret, "resmom": resmom, "beta": beta_out,
        "dollar_vol": (close * piv["volume"]).to_numpy(float),
        "bars": close.notna().cumsum().to_numpy(float),
        "min_bars": reg_len + 2,
    }


def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).last().values
            if v >= start_i]


# ── 종목 선정 ────────────────────────────────────────────────────────────
def _pool(P, i: int, cfg: dict) -> np.ndarray:
    """PRET>0 상위 90 — 모든 그룹의 공통 모집단."""
    px_i = P["close"][i]
    base = np.isfinite(px_i) & (P["bars"][i] >= P["min_bars"]) & np.isfinite(P["pret"][i])
    if not base.any():
        return np.array([], dtype=int)
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dv_med = np.nanmedian(dv, axis=0)
    univ = base & np.isfinite(dv_med)
    if not univ.any():
        return np.array([], dtype=int)
    keep = min(int(cfg["universe_n"]), int(univ.sum()))
    thr = np.sort(dv_med[univ])[::-1][keep - 1]
    univ &= dv_med >= thr
    cand = np.flatnonzero(univ & (P["pret"][i] > 0))
    if len(cand) == 0:
        return np.array([], dtype=int)
    tick = P["tickers"]
    return np.array(sorted(cand, key=lambda j: (-P["pret"][i][j], tick[j]))[:POOL_N],
                    dtype=int)


def targets(P, i: int, cfg: dict, spec: dict, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """(종목 인덱스, 비중) 반환. 비중 합 = 1."""
    pool = _pool(P, i, cfg)
    if len(pool) == 0:
        return np.array([], dtype=int), np.array([])
    tick, rs = P["tickers"], P["resmom"][i]
    top = pool[:HOLD_N] if len(pool) >= HOLD_N else pool
    kind = spec["kind"]

    if kind == "base":                                   # E — 베이스라인
        sel = top
    elif kind == "swap":                                 # 종목 k개 교체
        k = spec["k"]
        rest = pool[HOLD_N:]
        valid = np.array([j for j in rest if np.isfinite(rs[j])], dtype=int)
        if len(top) < HOLD_N or len(valid) < k:
            sel = top
        elif rng is not None:                            # D — 무작위 교체
            drop = set(int(x) for x in rng.choice(top, size=k, replace=False))
            add = rng.choice(valid, size=k, replace=False)
            sel = np.concatenate([[j for j in top if int(j) not in drop], add])
        else:
            key_lo = sorted(top, key=lambda j: (rs[j] if np.isfinite(rs[j]) else -np.inf,
                                                tick[j]))
            drop = set(int(x) for x in key_lo[:k])
            add = np.array(sorted(valid, key=lambda j: (-rs[j], tick[j]))[:k], dtype=int)
            sel = np.concatenate([[j for j in top if int(j) not in drop], add])
    elif kind == "tilt":                                 # 종목 고정, 비중만 기울임
        sel = top
        lam = spec["lam"]
        if len(sel) < 2:
            return sel, np.full(len(sel), 1.0 / max(len(sel), 1))
        v = np.array([rs[j] for j in sel], dtype=float)
        if not np.isfinite(v).any():
            return sel, np.full(len(sel), 1.0 / len(sel))
        # 유효값만 횡단면 순위 → [-1, +1] 등간격. NaN은 중립(0)
        z = np.zeros(len(sel))
        fin = np.isfinite(v)
        if fin.sum() >= 2:
            order = np.argsort(np.argsort(v[fin]))
            z[fin] = 2.0 * order / (fin.sum() - 1) - 1.0
        w = (1.0 + lam * z) / len(sel)
        return np.asarray(sel, dtype=int), w / w.sum()
    else:
        raise ValueError(kind)

    sel = np.asarray(sel, dtype=int)
    return sel, np.full(len(sel), 1.0 / len(sel))


# ── 검출력 사전 측정 (평균·t값을 계산하지 않는다) ────────────────────────
def _fwd_returns(P, i: int, j: int, sel: np.ndarray) -> np.ndarray:
    """리밸일 i→j 구간 총수익률 (비용·정수주 제외 — SE 추정용)."""
    c = P["close_ff"]
    with np.errstate(all="ignore"):
        return c[j][sel] / c[i][sel] - 1.0


def power_only(market: str, specs: list[dict], reg_len: int = REG_LEN) -> list[dict]:
    """각 spec의 A−E 월별 차분에 대해 **SE·MDE·겹침만** 산출한다.

    반환 dict에 평균·중위·t값·부호는 **담지 않는다**(사전등록 오염 방지).
    """
    P = build_panel(market, reg_len)
    cfg = MARKET_CFG[market]
    bmap = regime.bull_map(loading.load_bench(market), sma=SMA_BENCH)
    bull = np.array([bmap.get(d, False) for d in P["dates"]])
    start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= P["min_bars"])),
                SMA_BENCH, reg_len + 1)
    rebs = month_end_indices(P["dates"], start)
    out = []
    for spec in specs:
        diffs, overlaps, repl, n_bull = [], [], [], 0
        for a, b in zip(rebs[:-1], rebs[1:]):
            if not bull[a]:
                diffs.append(0.0)
                continue
            n_bull += 1
            se_, we_ = targets(P, a, cfg, {"kind": "base"})
            sa_, wa_ = targets(P, a, cfg, spec)
            if len(se_) == 0 or len(sa_) == 0:
                diffs.append(0.0)
                continue
            ra = _fwd_returns(P, a, b, sa_)
            re = _fwd_returns(P, a, b, se_)
            ma, me = np.isfinite(ra), np.isfinite(re)
            if not ma.any() or not me.any():
                diffs.append(0.0)
                continue
            va = float((wa_[ma] * ra[ma]).sum() / wa_[ma].sum())
            ve = float((we_[me] * re[me]).sum() / we_[me].sum())
            diffs.append((va - ve) * 100.0)
            inter = len(set(int(x) for x in sa_) & set(int(x) for x in se_))
            overlaps.append(inter / max(len(se_), 1))
            repl.append(len(se_) - inter)
        d_all = np.array(diffs, dtype=float)
        d_bull = np.array([x for x in diffs if x != 0.0], dtype=float)
        rec = {"spec": spec["label"], "n_all": len(d_all), "n_bull": n_bull,
               "overlap": float(np.mean(overlaps)) if overlaps else np.nan,
               "repl": float(np.mean(repl)) if repl else np.nan,
               "repl0": float(np.mean(np.array(repl) == 0)) if repl else np.nan}
        for tag, d in (("all", d_all), ("bull", d_bull)):
            se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
            rec[f"se_{tag}"] = se
            rec[f"mde_{tag}"] = K_MDE * se
        out.append(rec)
    return out


def cmd_power():
    specs = ([{"kind": "swap", "k": k, "label": f"교체 k={k}"} for k in (1, 2, 3, 5)]
             + [{"kind": "tilt", "lam": l, "label": f"비중틸트 λ={l}"}
                for l in (0.3, 0.5, 0.9)])
    print("=" * 96)
    print("[H-015 사전 검출력 측정] PROTOCOL §3.1-2 — SE·MDE만 산출 (평균·t값 미계산)")
    print(f"  MDE = {K_MDE:.6f} × SE  ·  e* = {E_STAR}%p/월  ·  e** = {E_STAR2}%p/월")
    print("=" * 96)
    for market in ("kr", "us"):
        rows = power_only(market, specs)
        print(f"\n── {market.upper()} ──")
        print(f"{'설계':<16}{'겹침':>8}{'교체':>7}{'강세달':>7}"
              f"{'SE(강세)':>10}{'MDE(강세)':>11}{'MDE(전체)':>11}  판정")
        for r in rows:
            v = ("검출가능(e**)" if r["mde_bull"] <= E_STAR2 else
                 "검출가능(e*)" if r["mde_bull"] <= E_STAR else "**측정불가**")
            print(f"{r['spec']:<16}{r['overlap']*100:>7.1f}%{r['repl']:>7.2f}"
                  f"{r['n_bull']:>7}{r['se_bull']:>10.4f}{r['mde_bull']:>11.3f}"
                  f"{r['mde_all']:>11.3f}  {v}")
    print("\n※ 이 출력에는 차분의 평균·t값·부호가 없다. 설계 선택이 결과에 오염되지"
          "\n  않도록 의도적으로 계산 경로를 두지 않았다(문서 §3.0).")


# ── 자체 검증 ────────────────────────────────────────────────────────────
def selftest():
    # ① β 소형 예제 (문서 §3.3 검증 2)
    rm = np.array([0.01, 0.02, 0.03])
    ri = np.array([0.02, 0.04, 0.06])
    n = 3
    cov = (ri * rm).sum() / n - ri.mean() * rm.mean()
    var = (rm * rm).sum() / n - rm.mean() ** 2
    beta = cov / var
    alpha = ri.mean() - beta * rm.mean()
    assert abs(beta - 2.0) < 1e-12, beta
    assert abs(alpha) < 1e-15, alpha
    assert np.allclose(ri - alpha - beta * rm, 0.0, atol=1e-15)

    # ② 항등식 Σ_W ε = MAR_W − (n_W/n_R)·MAR_R  (문서 §3.3 검증 4)
    #    계수는 231/750 상수가 아니라 n_W/n_R 이다 — 게이트 2회차 N4 정정
    rng = np.random.default_rng(7)
    nR, nW = 750, 231
    rm2 = rng.normal(0, 0.01, nR)
    ri2 = 0.0003 + 1.3 * rm2 + rng.normal(0, 0.015, nR)
    b = np.cov(ri2, rm2, ddof=1)[0, 1] / np.var(rm2, ddof=1)
    a = ri2.mean() - b * rm2.mean()
    W = slice(nR - nW, nR)
    lhs = float((ri2[W] - a - b * rm2[W]).sum())
    mar_W = float((ri2[W] - b * rm2[W]).sum())
    mar_R = float((ri2 - b * rm2).sum())
    assert abs(lhs - (mar_W - (nW / nR) * mar_R)) < 1e-10
    assert abs(nR * a - mar_R) < 1e-10          # 750·α̂ = MAR_R

    # ③ β=0 이면 총모멘텀 표준화값과 **같지 않다** (검증 5)
    ri3 = rng.normal(0.001, 0.02, nR)
    rm3 = rng.normal(0, 0.01, nR)
    b3 = np.cov(ri3, rm3, ddof=1)[0, 1] / np.var(rm3, ddof=1)
    a3 = ri3.mean() - b3 * rm3.mean()
    eps3 = ri3[W] - a3 - b3 * rm3[W]
    res_score = eps3.sum() / eps3.std(ddof=1)
    raw_score = ri3[W].sum() / ri3[W].std(ddof=1)
    assert abs(res_score - raw_score) > 1.0, (res_score, raw_score)

    # ④ 순수 베타 종목(α=0)의 RESMOM 부호는 동전던지기 (검증 6)
    hits = 0
    trials = 400
    for s in range(trials):
        g = np.random.default_rng(1000 + s)
        m = g.normal(0.0006, 0.01, nR)          # 강세장
        y = 1.6 * m + g.normal(0, 0.012, nR)    # α = 0
        bb = np.cov(y, m, ddof=1)[0, 1] / np.var(m, ddof=1)
        aa = y.mean() - bb * m.mean()
        hits += ((y[W] - aa - bb * m[W]).sum() > 0)
    rate = hits / trials
    assert 0.40 < rate < 0.60, rate            # 부호 필터 금지의 근거

    # ⑤ 틸트는 종목 집합을 바꾸지 않는다 → 겹침 100%
    P = {"tickers": np.array(["a", "b", "c", "d"]), "resmom": np.array([[3.0, 1.0, 2.0, 0.0]])}
    v = P["resmom"][0]
    order = np.argsort(np.argsort(v))
    z = 2.0 * order / (len(v) - 1) - 1.0
    w = (1.0 + 0.3 * z) / len(v)
    w = w / w.sum()
    assert abs(w.sum() - 1.0) < 1e-12
    assert w[0] > w[2] > w[1] > w[3], w         # RESMOM 높을수록 비중 큼
    assert abs(w.max() / w.min() - 1.3 / 0.7) < 1e-9

    # ⑥ MDE 상수
    assert abs(K_MDE - 2.801585) < 1e-5

    # ⑦ **척도 불변성** — 틸트 차분은 λ에 정확히 비례한다 (문서 §5.2).
    #    w = (1+λz)/n 이고 Σz = 0 이므로 Σw = 1 (재정규화 불필요).
    #    포트 수익 = (1/n)Σr + λ·(1/n)Σ z_j r_j  →  d_λ = λ · d_1 (정확히).
    #    따라서 SE도 λ배가 되고 **t는 λ에 완전히 불변**이다.
    #    ⇒ 처치를 줄여 MDE 문턱을 통과하는 것은 검출력 개선이 아니다.
    g = np.random.default_rng(11)
    r = g.normal(0.01, 0.08, (200, HOLD_N))
    zc = 2.0 * np.arange(HOLD_N) / (HOLD_N - 1) - 1.0
    assert abs(zc.sum()) < 1e-12
    d1 = None
    for lam in (0.3, 0.5, 0.9):
        wl = (1.0 + lam * zc) / HOLD_N
        assert abs(wl.sum() - 1.0) < 1e-12          # 재정규화 불필요
        d = r @ wl - r.mean(axis=1)
        if d1 is None:
            d1, lam1 = d, lam
        else:
            assert np.allclose(d, d1 * (lam / lam1), atol=1e-14)

    print("selftest: 7개 항목 통과 (β예제·항등식·β=0불일치·부호동전던지기·"
          "틸트비중·MDE상수·척도불변성)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--power":
        cmd_power()
    else:
        print(__doc__)
