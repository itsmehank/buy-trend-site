"""Phase 3 — US에서 EV 상승 원인 분해. 원시 vs S&P500 차감 vs 무작위 날짜 대조군."""
import numpy as np, pandas as pd
from pipeline import backtest, config, data, signals, universe

HOLDS, KEYS = config.HOLDS_MA, ["L|SMA|20", "S|SMA|20"]
rng = np.random.default_rng(20260729)
px = data.load_cache("us_test")
# 거래대금 필터는 이벤트 시점 기준으로 적용 (tail(20)은 "오늘" 기준 = look-ahead)
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker=='^GSPC'")
         [["date","close"]].sort_values("date").rename(columns={"close":"bench"}))

sig = {h: {"s": [], "b": []} for h in HOLDS}
rnd = {h: [] for h in HOLDS}
for t, g in px.groupby("ticker"):
    m = (g[["date","close","volume"]].sort_values("date")
         .merge(bench, on="date").reset_index(drop=True))
    if len(m) < 900: continue
    c, b = m["close"].to_numpy(float), m["bench"].to_numpy(float)
    if not np.all(c > 0): continue
    dv = (m["close"] * m["volume"]).rolling(20, min_periods=20).mean().to_numpy()
    liquid = np.nan_to_num(dv, nan=0.0) >= config.US_MIN_DOLLAR_VOL
    scan = signals.ma_signal_scan(m["close"].reset_index(drop=True))
    evs = [np.asarray(scan[k]["events"], dtype=int) for k in KEYS if len(scan[k]["events"])]
    if not evs: continue
    ev = np.unique(np.concatenate(evs))
    ev = ev[liquid[ev]]                       # 진입 시점 유동성 조건
    if not len(ev): continue
    fake = rng.integers(0, len(c), size=len(ev))
    fake = fake[liquid[fake]]                 # 대조군도 동일 조건
    for h in HOLDS:
        v = ev[ev + h < len(c)]
        if len(v):
            sig[h]["s"].append(c[v+h]/c[v] - 1.0); sig[h]["b"].append(b[v+h]/b[v] - 1.0)
        f = fake[fake + h < len(c)]
        if len(f):
            rnd[h].append((c[f+h]/c[f]) - (b[f+h]/b[f]))

def ev_of(r):
    r = r * 100; w, l = r[r > 0], r[r <= 0]
    if not len(l) or l.mean() == 0: return None, None, None
    p = len(w)/len(r)*100; aw = w.mean() if len(w) else 0.0; al = l.mean()
    return backtest.ev(p, aw, al), p, aw/abs(al)

print(f"{'보유':>6} | {'원시 EV':>8} {'승률':>7} | {'초과 EV':>8} {'승률':>7} {'손익비':>7} | {'무작위날짜 EV':>13}")
print("-"*76)
for h in HOLDS:
    s = np.concatenate(sig[h]["s"]); b = np.concatenate(sig[h]["b"])
    e1,p1,_ = ev_of(s); e2,p2,r2 = ev_of(s-b); e3,_,_ = ev_of(np.concatenate(rnd[h]))
    print(f"{h:>4}일 | {e1:>8.3f} {p1:>6.1f}% | {e2:>8.3f} {p2:>6.1f}% {r2:>7.2f} | {e3:>13.3f}")
print()
s2 = np.concatenate(sig[252]["s"]); b2 = np.concatenate(sig[252]["b"])
print(f"252일 기준 · 표본 {len(s2):,}건")
print(f"  원시 EV {ev_of(s2)[0]:.3f} → 지수 차감 후 {ev_of(s2-b2)[0]:.3f} "
      f"(드리프트 기여 {(1-ev_of(s2-b2)[0]/ev_of(s2)[0])*100:.0f}%)")
print(f"  신호 초과 EV {ev_of(s2-b2)[0]:.3f} vs 무작위 날짜 {ev_of(np.concatenate(rnd[252]))[0]:.3f}")
