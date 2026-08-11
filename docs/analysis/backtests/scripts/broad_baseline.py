"""베이스라인 C (넓은 유니버스) + 부품 3종 — H-020 / H-021 / H-022 공용.

베이스라인 C
  유니버스 : 상장 250거래일 이상 · 60일 거래대금 중앙값 상위 400(KR)/500(US)
  보유     : 유니버스 **전 종목 동일가중**
  판정·체결: 매월 마지막 거래일 T 종가 판정 → **T+1 종가 체결**
  레짐     : T에 지수>SMA200이면 신규 편입 허용. 아니면 신규 편입 없이 기존 보유 유지,
             유니버스 탈락 종목만 매도. **양 팔에 동일 적용**

부품 (각각이 하나의 가설)
  H-020 max   : MAX = 직전 21일 최대 일간수익률. 상위 10% 제외 후 나머지 균등 재배분
  H-021 beta  : β = 250일 지수 대비 OLS 기울기(0.2~3.0 클리핑). w ∝ 1/β, 종목당 상한 1%
  H-022 expo  : 총 주식 노출 e = min(1, (0.15/σ̂)²). σ̂ = 지수 21일 변동성 연율화

  PYTHONPATH=.:docs/analysis .venv/bin/python \
    docs/analysis/backtests/scripts/broad_baseline.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/broad_baseline.py --power

**--power 는 SE·MDE·겹침·처치비중만 출력한다. 평균·t값·부호는 계산 경로가 없다.**
사전등록 오염 방지를 위한 의도적 제약이다(PROTOCOL §3.1-2).
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading, regime

_N = NormalDist()
K_MDE = _N.inv_cdf(0.975) + _N.inv_cdf(0.80)      # 2.801585

MIN_BARS = 250
DV_WIN = 60
SMA_BENCH = 200
MAX_WIN = 21                  # H-020 MAX 창
MAX_FRAC = 0.10               # 상위 10% 제외
BETA_WIN = 250                # H-021 회귀 창
BETA_CLIP = (0.2, 3.0)
W_CAP = 0.01                  # 종목당 상한 1%
VOL_WIN = 21                  # H-022 지수 변동성 창
VOL_TARGET = 0.15

MARKET_CFG = {"kr": {"universe_n": 400}, "us": {"universe_n": 500}}


def _csum(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def build_panel(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx, cols = close.index, close.columns
    for k in piv:
        piv[k] = piv[k].reindex(index=idx, columns=cols)

    # PROTOCOL §1 — KR 거래정지 봉 OR 마스킹
    bad = (close <= 0) | (piv["open"] <= 0)
    if bad.to_numpy().any():
        for k in piv:
            piv[k] = piv[k].mask(bad)
        close = piv["close"]

    c = close.to_numpy(float)
    n_days, n_tk = c.shape
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0

    # MAX — 직전 21일 최대 일간수익률 (판단일 포함)
    mx = pd.DataFrame(ret).rolling(MAX_WIN).max().to_numpy(float)

    # 시장수익률
    bdf = loading.load_bench(market)
    bench = (pd.Series(bdf["close"].to_numpy(float), index=pd.to_datetime(bdf["date"]))
             .reindex(pd.to_datetime(idx)).ffill())
    bm = bench.to_numpy(float)
    rm = np.full(n_days, np.nan)
    rm[1:] = bm[1:] / bm[:-1] - 1.0

    # β — 250일 롤링 OLS (누적합)
    ok = np.isfinite(ret) & np.isfinite(rm)[:, None]
    R = np.where(ok, ret, 0.0)
    M = np.where(ok, np.broadcast_to(rm[:, None], c.shape), 0.0)
    C = {k: _csum(v) for k, v in
         {"n": ok.astype(float), "y": R, "x": M, "xy": R * M, "xx": M * M}.items()}
    beta = np.full_like(c, np.nan)
    for t in range(BETA_WIN, n_days):
        lo, hi = t - BETA_WIN + 1, t + 1
        nn = C["n"][hi] - C["n"][lo]
        sy, sx = C["y"][hi] - C["y"][lo], C["x"][hi] - C["x"][lo]
        sxy, sxx = C["xy"][hi] - C["xy"][lo], C["xx"][hi] - C["xx"][lo]
        with np.errstate(all="ignore"):
            cov = sxy / nn - (sx / nn) * (sy / nn)
            var = sxx / nn - (sx / nn) ** 2
            b = np.where((var > 0) & (nn >= BETA_WIN * 0.8), cov / var, np.nan)
        beta[t] = b

    # 종목 수준 추세·모멘텀 지표
    cdf = pd.DataFrame(c)
    sma200 = cdf.rolling(200).mean().to_numpy(float)
    with np.errstate(all="ignore"):
        above = c / sma200 - 1.0                      # 자기 SMA200 대비 위치
    far, near2 = np.full_like(c, np.nan), np.full_like(c, np.nan)
    far[252:], near2[21:] = c[:-252], c[:-21]
    with np.errstate(all="ignore"):
        mom12 = near2 / far - 1.0                     # 12−1 모멘텀
    hi52 = cdf.rolling(252).max().to_numpy(float)
    with np.errstate(all="ignore"):
        offhi = c / hi52 - 1.0                        # 52주 신고가 대비 (0 = 신고가)

    # 지수 21일 실현변동성 (연율)
    ivol = (pd.Series(rm).rolling(VOL_WIN).std().to_numpy(float) * np.sqrt(252))

    return {"dates": np.asarray(idx), "tickers": np.asarray(cols, dtype=str),
            "close": c, "close_ff": close.ffill().to_numpy(float),
            "max": mx, "beta": beta, "ivol": ivol,
            "above": above, "mom12": mom12, "offhi": offhi,
            "dollar_vol": (close * piv["volume"]).to_numpy(float),
            "bars": close.notna().cumsum().to_numpy(float)}


def month_end_indices(dates: np.ndarray, start_i: int) -> list[int]:
    s = pd.Series(np.arange(len(dates)), index=pd.to_datetime(dates))
    return [int(v) for v in s.groupby([s.index.year, s.index.month]).last().values
            if v >= start_i]


def universe(P, i: int, cfg: dict) -> np.ndarray:
    px = P["close"][i]
    base = np.isfinite(px) & (px > 0) & (P["bars"][i] >= MIN_BARS)
    if not base.any():
        return np.array([], dtype=int)
    dv = P["dollar_vol"][max(0, i - DV_WIN + 1):i + 1]
    with np.errstate(all="ignore"):
        dvm = np.nanmedian(dv, axis=0)
    u = base & np.isfinite(dvm)
    if not u.any():
        return np.array([], dtype=int)
    keep = min(int(cfg["universe_n"]), int(u.sum()))
    thr = np.sort(dvm[u])[::-1][keep - 1]
    u &= dvm >= thr
    cand = np.flatnonzero(u)
    tk = P["tickers"]
    return np.array(sorted(cand, key=lambda j: (-dvm[j], tk[j])), dtype=int)


def weights(P, i: int, sel: np.ndarray, part: str | None) -> np.ndarray:
    """부품을 적용한 목표 비중. 합 ≤ 1 (H-022는 현금 보유로 합 < 1)."""
    n = len(sel)
    if n == 0:
        return np.array([])
    w = np.full(n, 1.0 / n)
    if part is None:
        return w
    if part == "max":                                   # H-020
        mv = P["max"][i][sel]
        fin = np.isfinite(mv)
        if fin.sum() < n * 0.5:
            return w                                    # 정보 부족한 달은 처치 없음
        k = int(round(n * MAX_FRAC))
        if k < 1:
            return w
        # MAX 큰 순(=-mv 오름차순). NaN 은 +inf 로 두어 **맨 뒤**로 보낸다.
        # (-inf 로 두면 오름차순 맨 앞에 와서 NaN 종목이 먼저 제외된다 — 실제 버그였다)
        order = np.argsort(np.where(fin, -mv, np.inf), kind="stable")
        drop = set(order[:k].tolist())
        keep = np.array([j for j in range(n) if j not in drop])
        out = np.zeros(n)
        out[keep] = 1.0 / len(keep)
        return out
    if part in ("trend", "mom", "nh"):                  # 조건 미달 종목 제외형
        key = {"trend": "above", "mom": "mom12", "nh": "offhi"}[part]
        thr = {"trend": 0.0, "mom": 0.0, "nh": -0.25}[part]
        v = P[key][i][sel]
        # 지표가 NaN 인 종목은 **제외하지 않는다** — 정보 없음은 조건 미달이 아니다.
        # (H-020 §5.4 버그 #1과 같은 실패 모드. KR 정지봉 마스킹이 rolling 창으로
        #  전파돼 trend 제외분의 6.6% · nh 제외분의 9.1%가 "최근 거래정지"였다)
        ok_ = ~np.isfinite(v) | (v > thr)
        if ok_.sum() < max(20, n * 0.05):               # 남는 종목이 너무 적으면 처치 없음
            return w
        out = np.zeros(n)
        out[ok_] = 1.0 / ok_.sum()
        return out
    if part == "beta":                                  # H-021
        b = P["beta"][i][sel]
        b = np.where(np.isfinite(b), np.clip(b, *BETA_CLIP), np.nan)
        if np.isfinite(b).sum() < n * 0.5:
            return w
        inv = np.where(np.isfinite(b), 1.0 / b, np.nan)
        inv = np.where(np.isfinite(inv), inv, np.nanmedian(inv))
        out = inv / inv.sum()
        # 종목당 상한 1% — 초과분을 미달 종목에 비례 재배분 (반복)
        for _ in range(50):
            over = out > W_CAP
            if not over.any():
                break
            excess = (out[over] - W_CAP).sum()
            out[over] = W_CAP
            room = ~over
            if not room.any() or out[room].sum() <= 0:
                break
            out[room] += excess * out[room] / out[room].sum()
        return out / out.sum() if out.sum() > 0 else w
    if part == "expo":                                  # H-022
        sig = P["ivol"][i]
        if not np.isfinite(sig) or sig <= 0:
            return w
        e = min(1.0, (VOL_TARGET / sig) ** 2)
        return w * e
    raise ValueError(part)


def run_arm(P, bull, cfg, part: str | None, cost: float = 0.0) -> dict:
    """월별 총수익 계열과 보유 집합. 평균·t값은 산출하지 않는다."""
    start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                SMA_BENCH, BETA_WIN, 1)
    rebs = [r for r in month_end_indices(P["dates"], start) if r + 1 < len(P["dates"])]
    c = P["close_ff"]
    held: set[int] = set()
    prev: dict[int, float] = {}
    rets, net, holds, expo, turns = [], [], [], [], []
    for a, b in zip(rebs[:-1], rebs[1:]):
        u = universe(P, a, cfg)
        if len(u) == 0:
            rets.append(0.0)
            net.append(0.0)
            holds.append(set())
            continue
        if bull[a]:
            sel = u                                     # 신규 편입 허용
        else:
            uset = set(int(x) for x in u)
            sel = np.array(sorted(uset & held), dtype=int)   # 기존 보유 ∩ 유니버스
            if len(sel) == 0:
                sel = u
        w = weights(P, a, sel, part)
        held = set(int(x) for x in sel)
        holds.append(held)
        expo.append(float(w.sum()))
        # 단방향 회전율 = Σ|목표비중 − 직전 드리프트 비중| / 2
        cur = {int(j): float(x) for j, x in zip(sel, w)}
        keys = set(cur) | set(prev)
        turn = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys) / 2.0
        turns.append(turn)
        with np.errstate(all="ignore"):
            r = c[b + 1][sel] / c[a + 1][sel] - 1.0
        m = np.isfinite(r)
        gross = float((w[m] * r[m]).sum()) if m.any() else 0.0
        rets.append(gross * 100.0)
        net.append((gross - turn * 2.0 * cost) * 100.0)      # 매도+매수 = 2×단방향
        # 다음 달 시작 비중 = 수익 반영 후 드리프트. **현금(1−Σw)을 이월**하고
        # 총자산으로 정규화해 합이 1이 되게 한다(H-023 검토 P5).
        if m.any():
            gw = w[m] * (1.0 + r[m])
            cash = 1.0 - float(w.sum())
            total = float(gw.sum()) + cash
            if total > 0:
                prev = {int(j): float(x) / total
                        for j, x in zip(np.asarray(sel)[m], gw)}
            else:
                prev = cur
        else:
            prev = cur
    return {"rets": np.array(rets), "net": np.array(net), "holds": holds,
            "turn": float(np.mean(turns)) if turns else np.nan,
            "expo": float(np.mean(expo)) if expo else np.nan, "n": len(rets)}


#: 부품별 처치 비중 산출 방식 — 척도 불변 기준(PROTOCOL §3.1-3)의 분모
def treat_frac(part: str, base: dict, arm: dict, P=None) -> float:
    if part == "max":
        return MAX_FRAC                                  # 제외 슬리브 비중 = 10%
    if part == "beta":                                   # 액티브 셰어
        return np.nan                                    # run_power 에서 실측
    if part == "expo":
        return 1.0 - arm["expo"]                         # 평균 현금 비중
    raise ValueError(part)


def run_power(market: str) -> list[dict]:
    P = build_panel(market)
    cfg = MARKET_CFG[market]
    bmap = regime.bull_map(loading.load_bench(market), sma=SMA_BENCH)
    bull = np.array([bmap.get(d, False) for d in P["dates"]])
    base = run_arm(P, bull, cfg, None)
    out = []
    for part, hid in (("max", "H-020"), ("beta", "H-021"), ("expo", "H-022"),
                      ("trend", "후보-추세"), ("mom", "후보-모멘텀"), ("nh", "후보-낙폭")):
        arm = run_arm(P, bull, cfg, part)
        d = arm["rets"] - base["rets"]
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
        ov = [len(a & e) / max(len(e), 1) for a, e in zip(arm["holds"], base["holds"]) if e]
        if part in ("trend", "mom", "nh"):               # 평균 제외 비중 실측
            fr = []
            start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                        SMA_BENCH, BETA_WIN, 1)
            for a in month_end_indices(P["dates"], start)[:-1]:
                u = universe(P, a, cfg)
                if len(u) == 0:
                    continue
                wa = weights(P, a, u, part)
                fr.append(float((wa == 0).sum() / len(u)))
            frac = float(np.mean(fr)) if fr else np.nan
        elif part == "beta":                             # 액티브 셰어 실측
            acts = []
            start = max(int(np.argmax(np.nanmax(P["bars"], axis=1) >= MIN_BARS)),
                        SMA_BENCH, BETA_WIN, 1)
            for a in month_end_indices(P["dates"], start)[:-1]:
                u = universe(P, a, cfg)
                if len(u) == 0:
                    continue
                acts.append(float(np.abs(weights(P, a, u, "beta")
                                         - weights(P, a, u, None)).sum() / 2))
            frac = float(np.mean(acts)) if acts else np.nan
        else:
            frac = treat_frac(part, base, arm)
        out.append({"id": hid, "part": part, "n": len(d), "se": se,
                    "mde": K_MDE * se, "overlap": float(np.mean(ov)) if ov else np.nan,
                    "frac": frac, "need": K_MDE * se / frac if frac and frac > 0 else np.nan})
    return out


def cmd_power():
    print("=" * 100)
    print("[베이스라인 C 사전 검출력] PROTOCOL §3.1-2 — SE·MDE만 산출 (평균·t값 미계산)")
    print(f"  MDE = {K_MDE:.6f} × SE  ·  필요 효과 = MDE ÷ 처치 비중 (척도 불변, §3.1-3)")
    print("=" * 100)
    for m in ("kr", "us"):
        print(f"\n── {m.upper()} ──")
        print(f"{'가설':<8}{'부품':<7}{'달':>5}{'겹침':>8}{'처치비중':>10}"
              f"{'SE':>9}{'MDE(월%p)':>11}{'필요효과(월%p)':>15}")
        for r in run_power(m):
            print(f"{r['id']:<8}{r['part']:<7}{r['n']:>5}{r['overlap']*100:>7.1f}%"
                  f"{r['frac']*100:>9.1f}%{r['se']:>9.4f}{r['mde']:>11.4f}"
                  f"{r['need']:>15.3f}")
    print("\n※ '필요효과' = 처치가 작용하는 부분이 매달 이만큼 벌어져야 검출된다."
          "\n  H-020은 제외 슬리브 vs 잔여, H-021은 액티브 부분, H-022는 현금화 부분 기준.")


PARTS = (("max", "MAX 상위10% 제외", 0.0754),
         ("beta", "저베타 1/β 기울임", None),
         ("expo", "실현분산 노출조절", None),
         ("trend", "자기 SMA200 하회 제외", None),
         ("mom", "12−1 모멘텀 ≤0 제외", None),
         ("nh", "52주고점 −25% 하회 제외", None))


def cmd_run():
    """**측정 모드** — 각 부품의 효과 추정치와 95% CI를 낸다.

    이것은 채택/기각 검정이 아니다. 검출력이 부족함은 --power 로 이미 확인됐다
    (실효 검정력 1.3~29%). 따라서 **부호나 유의성으로 결론을 내리지 않는다.**
    산출물은 추정치와 그 불확실성이며, 향후 메타분석의 입력이다.
    """
    print("=" * 104)
    print("[베이스라인 C 부품별 효과 측정] — 검정이 아니라 추정. 채택/기각을 하지 않는다.")
    print("  비용 반영(편도 KR 0.14% / US 0.05%) · 월 %p · CI는 ±1.96×SE")
    print("=" * 104)
    for m in ("kr", "us"):
        cost = 0.0014 if m == "kr" else 0.0005
        P = build_panel(m)
        cfg = MARKET_CFG[m]
        bmap = regime.bull_map(loading.load_bench(m), sma=SMA_BENCH)
        bull = np.array([bmap.get(d, False) for d in P["dates"]])
        base = run_arm(P, bull, cfg, None, cost)
        print(f"\n── {m.upper()} ──  {base['n']}개월 · 대조군 회전율 {base['turn']*100:.2f}%/월")
        print(f"{'부품':<26}{'평균':>9}{'중위':>9}{'95% CI':>22}{'t':>7}"
              f"{'CI가 0 제외':>12}")
        for part, label, _ in PARTS:
            arm = run_arm(P, bull, cfg, part, cost)
            d = arm["net"] - base["net"]
            n = len(d)
            mu = float(np.mean(d))
            med = float(np.median(d))
            se = float(np.std(d, ddof=1) / np.sqrt(n))
            lo, hi = mu - 1.959964 * se, mu + 1.959964 * se
            t = mu / se if se > 0 else np.nan
            excl = "제외" if lo * hi > 0 else "포함"
            print(f"{label:<26}{mu:>+9.4f}{med:>+9.4f}"
                  f"   [{lo:>+7.4f}, {hi:>+7.4f}]{t:>7.2f}{excl:>12}")
    print("\n※ 이 표는 추정치다. --power 가 보인 대로 실효 검정력이 1.3~29%이므로"
          "\n  **CI가 0을 포함한다는 사실은 '효과 없음'의 증거가 아니다.**"
          "\n  부호가 시장 간 일치하는지, 문헌 값과 자릿수가 맞는지만 읽는다.")


def cmd_verdict():
    """**통합 표본 판정** (PROTOCOL §3 개정 2026-08-11, 사용자 승인).

    판정 규칙 — 양 시장 교집합이 아니라 **월별 차분을 이어붙인 통합 표본**.
    시장별 추정치는 임계 없이 병기한다(과적합 방어 대체 장치).
    """
    print("=" * 100)
    print("[통합 표본 판정] PROTOCOL §3 개정본 · 검정력 기준 50%")
    print("  주 판정 = 통합 표본 · 시장별은 임계 없이 병기(과적합 확인용)")
    print("=" * 100)
    series = {}
    for m in ("kr", "us"):
        cost = 0.0014 if m == "kr" else 0.0005
        P = build_panel(m)
        cfg = MARKET_CFG[m]
        bmap = regime.bull_map(loading.load_bench(m), sma=SMA_BENCH)
        bull = np.array([bmap.get(d, False) for d in P["dates"]])
        base = run_arm(P, bull, cfg, None, cost)
        for part, label, _ in PARTS:
            arm = run_arm(P, bull, cfg, part, cost)
            series.setdefault(part, {})[m] = arm["net"] - base["net"]
    k = len(PARTS)
    crit = _N.inv_cdf(1 - 0.05 / (2 * k))
    print(f"\n다중비교: {k}개 부품 · Bonferroni α=0.05/{k} → |t| > {crit:.3f}\n")
    print(f"{'부품':<26}{'통합 평균':>10}{'통합 t':>9}{'KR':>9}{'US':>9}"
          f"{'부호일치':>9}  판정")
    for part, label, _ in PARTS:
        d = np.concatenate([series[part]["kr"], series[part]["us"]])
        mu = float(np.mean(d))
        t = mu / (np.std(d, ddof=1) / np.sqrt(len(d)))
        mk, mu_ = float(np.mean(series[part]["kr"])), float(np.mean(series[part]["us"]))
        agree = "✓" if (mk > 0) == (mu_ > 0) else "✗"
        if abs(t) > crit:
            v = "**채택**" if t > 0 else "**기각**(문헌 반대)"
        else:
            v = "측정 불가"
        print(f"{label:<26}{mu:>+10.4f}{t:>9.2f}{mk:>+9.4f}{mu_:>+9.4f}{agree:>9}  {v}")
    print("\n※ 비유의는 '측정 불가'다. **'효과 없음'이 아니다**(PROTOCOL §3.1-7-나).")


def selftest():
    assert abs(K_MDE - 2.801585) < 1e-6

    # ① 베이스라인 비중 = 동일가중, 합 1
    P = {"max": np.array([[0.01, 0.09, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]]),
         "beta": np.array([[1.0] * 10]), "ivol": np.array([0.15])}
    sel = np.arange(10)
    w0 = weights(P, 0, sel, None)
    assert abs(w0.sum() - 1.0) < 1e-12 and np.allclose(w0, 0.1)

    # ② H-020 — MAX 상위 10%(1종목) 제외, 나머지 균등, 합 1
    w1 = weights(P, 0, sel, "max")
    assert abs(w1.sum() - 1.0) < 1e-12
    assert w1[9] == 0.0, w1                      # MAX 0.10 이 최대 → 제외
    assert np.allclose(w1[[0, 1, 2]], 1 / 9)
    assert (w1 > 0).sum() == 9

    # ③ H-020 — 제외 종목 수가 정확히 10%
    P2 = {"max": np.array([np.linspace(0.01, 0.5, 100)]), "beta": np.array([[1.0] * 100]),
          "ivol": np.array([0.15])}
    w2 = weights(P2, 0, np.arange(100), "max")
    assert (w2 == 0).sum() == 10 and abs(w2.sum() - 1.0) < 1e-12

    # ④ H-021 — β 낮은 종목이 더 큰 비중, 상한 1% 준수, 합 1
    P3 = {"max": np.zeros((1, 200)), "ivol": np.array([0.15]),
          "beta": np.array([np.linspace(0.3, 2.5, 200)])}
    w3 = weights(P3, 0, np.arange(200), "beta")
    assert abs(w3.sum() - 1.0) < 1e-9, w3.sum()
    assert w3.max() <= W_CAP + 1e-9, w3.max()
    assert w3[0] >= w3[-1]                        # β 낮을수록 비중 크다

    # ⑤ H-022 — 노출 = min(1, (0.15/σ)²)
    for sig, want in ((0.15, 1.0), (0.30, 0.25), (0.10, 1.0), (0.2121320, 0.5)):
        Pv = {"max": np.zeros((1, 4)), "beta": np.array([[1.0] * 4]),
              "ivol": np.array([sig])}
        wv = weights(Pv, 0, np.arange(4), "expo")
        assert abs(wv.sum() - want) < 1e-4, (sig, wv.sum(), want)

    # ⑥ 척도 불변 — 필요 효과 = MDE / 처치비중, 처치가 작으면 필요 효과가 커진다
    assert (K_MDE * 0.02) / 0.05 > (K_MDE * 0.02) / 0.20

    # ⑦ H-020 차분 항등식: d = frac × (잔여평균 − 제외평균)
    r = np.array([0.05, -0.02, 0.10, 0.01, 0.03, -0.04, 0.06, 0.02, -0.01, 0.08])
    e_ret = r.mean()
    a_ret = float((w1 * r).sum())
    keep_m = r[w1 > 0].mean()
    excl_m = r[w1 == 0].mean()
    assert abs((a_ret - e_ret) - 0.1 * (keep_m - excl_m)) < 1e-12

    # ⑧ 회전율 정의 — Σ|Δw|/2 (단방향). 손계산 대조
    cur = {0: 0.5, 1: 0.5}
    pv = {0: 0.6, 1: 0.4}
    t = sum(abs(cur.get(k, 0.0) - pv.get(k, 0.0)) for k in set(cur) | set(pv)) / 2
    assert abs(t - 0.10) < 1e-12, t                      # 0.1 매도 + 0.1 매수 = 단방향 0.1
    pv2 = {}                                             # 첫 달: 현금 → 전액 매수
    t2 = sum(abs(cur.get(k, 0.0) - pv2.get(k, 0.0)) for k in set(cur) | set(pv2)) / 2
    assert abs(t2 - 0.50) < 1e-12, t2                    # Σ|w|/2 = 0.5 = 편도 1회분

    # ⑨ 드리프트 정규화 — 현금 이월 후 합 1
    w_ = np.array([0.3, 0.3]); r_ = np.array([0.10, -0.10])
    gw = w_ * (1 + r_); cash = 1.0 - w_.sum(); total = gw.sum() + cash
    assert abs((gw / total).sum() + cash / total - 1.0) < 1e-12
    assert abs(cash - 0.4) < 1e-12                       # 현금이 소멸하지 않는다

    # ⑩ NaN 은 제외형 부품에서 배제되지 않는다 (P4 회귀 방지)
    av = np.concatenate([np.full(50, 0.1), [np.nan], np.full(49, -0.1)])
    Pn = {"above": av[None, :], "mom12": np.zeros((1, 100)),
          "offhi": np.zeros((1, 100)), "max": np.zeros((1, 100)),
          "beta": np.ones((1, 100)), "ivol": np.array([0.15])}
    wn = weights(Pn, 0, np.arange(100), "trend")
    assert wn[50] > 0, "NaN 종목이 제외됐다"          # 정보 없음 ≠ 조건 미달
    assert wn[51] == 0.0, "조건 미달 종목이 남았다"    # above=-0.1 → 제외
    assert wn[0] > 0 and (wn > 0).sum() == 51        # 통과 50 + NaN 1

    print("selftest: 10개 항목 통과 (동일가중·MAX제외·10%정확·베타틸트상한·노출식·"
          "척도불변·차분항등식·회전율정의·드리프트정규화·NaN유지)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--power":
        cmd_power()
    elif arg == "--run":
        cmd_run()
    elif arg == "--verdict":
        cmd_verdict()
    else:
        print(__doc__)
