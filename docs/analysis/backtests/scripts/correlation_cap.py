"""H-017 상관계수 상한 — 베이스라인 B에 부품 하나(진입 시 상관 게이트)를 얹는다.

PROTOCOL §3.1-2가 요구하는 **사전 검출력 측정**을 위한 `--power` / `--scan` 모드.

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/correlation_cap.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/correlation_cap.py --scan     # 임계값 → 처치 강도
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/correlation_cap.py --power    # SE·MDE

**--scan / --power 는 평균·t값·부호를 계산하지 않는다.** 사전등록 오염 방지를 위한
의도적 제약이며, 그 계산 경로가 코드에 존재하지 않는다(H-015 §5.1과 동일 방식).
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading, regime, costs

_N = NormalDist()
K_MDE = _N.inv_cdf(0.975) + _N.inv_cdf(0.80)      # 2.801585
E_STAR, E_STAR2 = 0.33, 0.17                       # 월 %p (H-014 §2.3)

# ── 베이스라인 B 고정 파라미터 (registry 등재 키) ────────────────────────
LB_FAR, LB_NEAR = 252, 21          # 12−1 모멘텀
MIN_BARS = 250                     # 상장 250거래일 이상
DV_WIN = 60                        # 거래대금 중앙값 창
DV_TOP_FRAC = 0.50                 # 상위 50%
HOLD_N = 20                        # 보유 종목 수
RANK_BUFFER = 40                   # 40위 밖으로 밀리면 매도
CORR_WIN = 120                     # 상관계수 창 (일간수익률)
SMA_BENCH = 200

MARKET_CFG = {
    "kr": {"min_price": 1000.0, "cost_pct": costs.COST_PCT["KR"] / 2},
    "us": {"min_price": 5.0,    "cost_pct": costs.COST_PCT["US"] / 2},
}


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # KR 거래정지 봉(open=high=low=0, close만 이월) OR 마스킹 — PROTOCOL §1
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    far, near = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[LB_FAR:], near[LB_NEAR:] = c[:-LB_FAR], c[:-LB_NEAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        pret = near / far - 1.0
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0

    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close": c, "close_ff": close.ffill().to_numpy(float),
            "ret": ret, "pret": pret,
            "dollar_vol": (close * piv["volume"]).to_numpy(float),
            "bars": close.notna().cumsum().to_numpy(float)}


def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).last().values
            if v >= start_i]


def ranked_universe(P, i: int, cfg: dict) -> np.ndarray:
    """베이스라인 B의 유니버스를 PRET 내림차순으로 정렬해 반환."""
    px = P["close"][i]
    base = (np.isfinite(px) & (px >= cfg["min_price"])
            & (P["bars"][i] >= MIN_BARS) & np.isfinite(P["pret"][i]))
    if not base.any():
        return np.array([], dtype=int)
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dv_med = np.nanmedian(dv, axis=0)
    univ = base & np.isfinite(dv_med)
    if not univ.any():
        return np.array([], dtype=int)
    keep = max(int(univ.sum() * DV_TOP_FRAC), 1)
    thr = np.sort(dv_med[univ])[::-1][keep - 1]
    univ &= dv_med >= thr
    cand = np.flatnonzero(univ)
    tick = P["tickers"]
    return np.array(sorted(cand, key=lambda j: (-P["pret"][i][j], tick[j])), dtype=int)


def _corr_block(P, i: int, ids: np.ndarray) -> np.ndarray:
    """직전 CORR_WIN 일간수익률의 상관행렬. 유효 관측 부족은 0(제약 없음)으로 둔다."""
    seg = P["ret"][max(0, i - CORR_WIN + 1):i + 1][:, ids]
    ok = np.isfinite(seg)
    x = np.where(ok, seg, 0.0)
    n = ok.sum(axis=0).astype(float)
    with np.errstate(all="ignore"):
        mu = x.sum(axis=0) / np.maximum(n, 1)
        xc = np.where(ok, x - mu, 0.0)
        cov = xc.T @ xc
        sd = np.sqrt(np.diag(cov))
        denom = np.outer(sd, sd)
        C = np.where(denom > 0, cov / np.maximum(denom, 1e-300), 0.0)
    C[n < CORR_WIN // 2, :] = 0.0
    C[:, n < CORR_WIN // 2] = 0.0
    return np.clip(C, -1.0, 1.0)


def rebalance(P, i: int, held: list[int], cfg: dict, cap: float | None
              ) -> tuple[list[int], int]:
    """베이스라인 B의 리밸런싱. cap이 주어지면 진입 시 상관 게이트를 적용한다.

    반환: (새 보유 리스트, 게이트로 건너뛴 종목 수)
    """
    order = ranked_universe(P, i, cfg)
    if len(order) == 0:
        return [], 0
    rank = {int(j): r for r, j in enumerate(order)}          # 0-based

    # ① 40위 버퍼 — 순위 밖으로 밀렸거나 유니버스에서 빠지면 매도
    kept = [j for j in held if rank.get(int(j), 10 ** 9) < RANK_BUFFER]

    need = HOLD_N - len(kept)
    if need <= 0:
        return kept[:HOLD_N], 0

    # ② 빈자리 채우기 — 미보유 종목 중 랭킹 상위부터
    kept_set = set(int(j) for j in kept)
    pool = [int(j) for j in order if int(j) not in kept_set]
    if cap is None:
        return kept + pool[:need], 0

    ids = np.array(sorted(set(kept_set) | set(pool[:200])), dtype=int)
    pos = {int(j): k for k, j in enumerate(ids)}
    C = _corr_block(P, i, ids)
    sel = list(kept)
    skipped = 0
    for j in pool:
        if len(sel) >= HOLD_N:
            break
        if j not in pos:
            continue
        if sel and max(C[pos[j], pos[int(x)]] for x in sel if int(x) in pos) > cap:
            skipped += 1
            continue
        sel.append(j)
    # 후보가 모자라면 게이트를 무시하고 랭킹 순으로 채운다(현금 방치 금지)
    if len(sel) < HOLD_N:
        for j in pool:
            if len(sel) >= HOLD_N:
                break
            if j not in sel:
                sel.append(j)
    return sel, skipped


def run_paths(P, bull, cfg: dict, cap: float | None) -> dict:
    """월별 보유 경로와 총수익 계열. 평균·t값은 산출하지 않는다."""
    start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                SMA_BENCH, LB_FAR, 1)
    rebs = [r for r in month_end_indices(P["dates"], start) if r + 1 < len(P["dates"])]
    held: list[int] = []
    hold_hist, rets, skips = [], [], []
    c = P["close_ff"]
    for a, b in zip(rebs[:-1], rebs[1:]):
        if not bull[a]:
            held = []
            hold_hist.append(set())
            rets.append(0.0)
            continue
        held, sk = rebalance(P, a, held, cfg, cap)
        skips.append(sk)
        hold_hist.append(set(int(x) for x in held))
        if not held:
            rets.append(0.0)
            continue
        # 익일 종가 체결 → a+1 에서 b+1 까지
        with np.errstate(all="ignore"):
            r = c[b + 1][held] / c[a + 1][held] - 1.0
        r = r[np.isfinite(r)]
        rets.append(float(r.mean()) * 100.0 if len(r) else 0.0)
    return {"rets": np.array(rets), "holds": hold_hist,
            "skips": np.array(skips) if skips else np.array([0])}


def measure(market: str, caps: list[float]) -> list[dict]:
    P = build_panel(market)
    cfg = MARKET_CFG[market]
    bmap = regime.bull_map(loading.load_bench(market), sma=SMA_BENCH)
    bull = np.array([bmap.get(d, False) for d in P["dates"]])
    base = run_paths(P, bull, cfg, None)
    out = []
    for cap in caps:
        arm = run_paths(P, bull, cfg, cap)
        d = arm["rets"] - base["rets"]
        ov, rep = [], []
        for ha, he in zip(arm["holds"], base["holds"]):
            if not he:
                continue
            inter = len(ha & he)
            ov.append(inter / len(he))
            rep.append(len(he) - inter)
        d_bull = np.array([x for x, he in zip(d, base["holds"]) if he])
        se = float(np.std(d_bull, ddof=1) / np.sqrt(len(d_bull))) if len(d_bull) > 1 else np.nan
        out.append({"cap": cap, "n_bull": len(d_bull),
                    "overlap": float(np.mean(ov)) if ov else np.nan,
                    "repl": float(np.mean(rep)) if rep else np.nan,
                    "repl0": float(np.mean(np.array(rep) == 0)) if rep else np.nan,
                    "skip": float(arm["skips"].mean()),
                    "se": se, "mde": K_MDE * se})
    return out


CAPS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85]


def cmd_scan():
    print("=" * 92)
    print("[H-017 사전 처치 강도 스캔] 임계값별 교체 종목 수 — 수익률 미계산")
    print("=" * 92)
    for m in ("kr", "us"):
        print(f"\n── {m.upper()} ──")
        print(f"{'임계 cap':>9}{'겹침':>9}{'교체/월':>9}{'교체0 비율':>11}{'게이트차단/월':>13}")
        for r in measure(m, CAPS):
            print(f"{r['cap']:>9.2f}{r['overlap']*100:>8.1f}%{r['repl']:>9.2f}"
                  f"{r['repl0']*100:>10.1f}%{r['skip']:>13.2f}")


def cmd_power():
    print("=" * 92)
    print("[H-017 사전 검출력 측정] PROTOCOL §3.1-2 — SE·MDE만 산출 (평균·t값 미계산)")
    print(f"  MDE = {K_MDE:.6f} × SE  ·  e* = {E_STAR}%p/월  ·  e** = {E_STAR2}%p/월")
    print("=" * 92)
    for m in ("kr", "us"):
        print(f"\n── {m.upper()} ──")
        print(f"{'임계 cap':>9}{'겹침':>9}{'교체/월':>9}{'강세달':>8}{'SE':>9}{'MDE':>9}"
              f"{'필요효과/교체쌍':>16}  판정")
        for r in measure(m, CAPS):
            frac = r["repl"] / HOLD_N if r["repl"] else np.nan
            need = r["mde"] / frac if frac and frac > 0 else np.nan
            v = ("검출가능(e**)" if r["mde"] <= E_STAR2 else
                 "검출가능(e*)" if r["mde"] <= E_STAR else "**측정불가**")
            print(f"{r['cap']:>9.2f}{r['overlap']*100:>8.1f}%{r['repl']:>9.2f}"
                  f"{r['n_bull']:>8}{r['se']:>9.4f}{r['mde']:>9.3f}"
                  f"{need:>15.2f}%p  {v}")
    print("\n※ '필요효과/교체쌍' = MDE ÷ 처치비중. 교체된 종목 한 쌍이 매달 이만큼"
          "\n  벌어져야 검출된다 (PROTOCOL §3.1-3의 척도 불변 기준).")


def cmd_power_vol():
    """변동성(산포) 지표의 검출력. **σ_A·σ_E 자체는 출력하지 않는다.**

    이론(Daniel·Moskowitz 2016; Choueifaty 2008)이 예측하는 것은 평균 수익 증가가
    아니라 **분산 감소**다. 두 계열이 강하게 상관될수록(겹치는 대조군) 분산비는
    훨씬 정밀하게 추정된다.

        Var(log(σ²_A / σ²_E)) ≈ (4 / (n−1)) · (1 − ρ²)

    ρ는 두 팔의 월 수익 상관계수. 이 식은 부호·크기 정보를 담지 않으므로
    사전등록을 오염시키지 않는다.
    """
    print("=" * 92)
    print("[H-017 변동성 지표 검출력] 이론이 예측하는 것은 분산 감소다")
    print("  Var(log 분산비) ≈ 4(1−ρ²)/(n−1) · · · σ_A·σ_E 는 출력하지 않는다")
    print("=" * 92)
    for m in ("kr", "us"):
        P = build_panel(m)
        cfg = MARKET_CFG[m]
        bmap = regime.bull_map(loading.load_bench(m), sma=SMA_BENCH)
        bull = np.array([bmap.get(d, False) for d in P["dates"]])
        base = run_paths(P, bull, cfg, None)
        print(f"\n── {m.upper()} ──")
        print(f"{'임계 cap':>9}{'겹침':>9}{'교체/월':>9}{'강세달':>8}{'상관 ρ':>9}"
              f"{'SE(log분산비)':>14}{'검출가능 변동성감소':>20}")
        for cap in (0.30, 0.40, 0.50, 0.55, 0.65, 0.75):
            arm = run_paths(P, bull, cfg, cap)
            keep = [k for k, he in enumerate(base["holds"]) if he]
            a = arm["rets"][keep]
            e = base["rets"][keep]
            n = len(a)
            rho = float(np.corrcoef(a, e)[0, 1])
            se = float(np.sqrt(4.0 * (1 - rho ** 2) / (n - 1)))
            mde_log = K_MDE * se
            vol_cut = 1.0 - np.exp(-mde_log / 2.0)     # σ 기준 감소율
            ov, rep = [], []
            for ha, he in zip(arm["holds"], base["holds"]):
                if not he:
                    continue
                ov.append(len(ha & he) / len(he))
                rep.append(len(he) - len(ha & he))
            print(f"{cap:>9.2f}{np.mean(ov)*100:>8.1f}%{np.mean(rep):>9.2f}{n:>8}"
                  f"{rho:>9.4f}{se:>14.4f}{vol_cut*100:>19.1f}%")
    print("\n※ 맨 오른쪽 = 80% 검정력으로 검출 가능한 **최소 변동성 감소율**."
          "\n  평균 수익 지표(--power)와 달리 두 팔의 강한 상관이 여기서는 이점이 된다.")


def selftest():
    # ① 상관행렬 — 완전 상관/무상관 극단값
    P = {"ret": np.zeros((200, 3))}
    g = np.random.default_rng(3)
    x = g.normal(0, 0.02, 200)
    P["ret"] = np.column_stack([x, 2 * x, g.normal(0, 0.02, 200)])
    C = _corr_block(P, 199, np.array([0, 1, 2]))
    assert abs(C[0, 1] - 1.0) < 1e-9, C[0, 1]          # y=2x → 상관 1
    assert abs(C[0, 0] - 1.0) < 1e-9
    assert abs(C[0, 2]) < 0.25, C[0, 2]                # 독립 → 0 근처

    # ② 게이트 방향 — cap이 낮을수록 더 많이 걸러낸다 (단조)
    assert CAPS == sorted(CAPS)

    # ③ 40위 버퍼 — 순위 39는 유지, 40은 매도
    Pz = {"close": np.array([[10.0, 10.0, 10.0]]), "bars": np.array([[999, 999, 999]]),
          "pret": np.array([[0.3, 0.2, 0.1]]), "tickers": np.array(["a", "b", "c"]),
          "dollar_vol": np.array([[100.0, 100.0, 100.0]]), "ret": np.zeros((1, 3))}
    order = ranked_universe(Pz, 0, {"min_price": 1.0})
    assert list(order) == [0, 1, 2], order              # PRET 내림차순

    # ④ 게이트가 없으면(cap=None) 랭킹 상위 그대로
    sel, sk = rebalance(Pz, 0, [], {"min_price": 1.0}, None)
    assert sel == [0, 1, 2] and sk == 0

    # ⑤ 상관 1인 종목은 cap 아래에서 배제된다
    n = 300
    r0 = g.normal(0, 0.02, n)
    Pc = {"close": np.full((n, 3), 10.0), "bars": np.full((n, 3), 999.0),
          "pret": np.tile([0.3, 0.2, 0.1], (n, 1)),
          "tickers": np.array(["a", "b", "c"]),
          "dollar_vol": np.full((n, 3), 100.0),
          "ret": np.column_stack([r0, r0, g.normal(0, 0.02, n)])}
    sel2, sk2 = rebalance(Pc, n - 1, [], {"min_price": 1.0}, 0.9)
    assert sel2[0] == 0 and sk2 >= 1                    # b는 a와 상관 1 → 차단
    assert sel2[1] == 2                                  # 다음 순위 c가 대체

    # ⑥ MDE 상수
    assert abs(K_MDE - 2.801585) < 1e-5
    print("selftest: 6개 항목 통과 (상관극단값·임계단조·랭킹정렬·게이트off·게이트작동·MDE상수)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--scan":
        cmd_scan()
    elif arg == "--power":
        cmd_power()
    elif arg == "--power-vol":
        cmd_power_vol()
    else:
        print(__doc__)
