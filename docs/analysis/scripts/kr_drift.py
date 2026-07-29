"""결정 2 — 252일 편중이 '시장 드리프트' 때문인지 실측.

방법: 같은 신호·같은 보유기간에 대해
  원시 승률 = P(종목 수익 > 0)
  초과 승률 = P(종목 수익 > 같은 창의 지수 수익)
원시는 보유기간에 따라 오르는데 초과는 평평하면, 상승분은 시장이 올려준 것이다.

한계: 상장폐지 종목이 캐시에 없어 생존 편향은 제거하지 못한다(드리프트만 분리).
      로컬에 US 가격이 없어 KR(2014~2026)만 측정한다.
"""
import numpy as np
import pandas as pd

from pipeline import config, signals

HOLDS = config.HOLDS_MA
KEYS = ["L|SMA|20", "S|SMA|20"]

px = pd.read_parquet("cache/prices_kr.parquet")
bench = (pd.read_parquet("cache/prices_bench.parquet")
         .query("ticker == '^KS11'")[["date", "close"]]
         .rename(columns={"close": "bench"}))

# 기간별로 (종목수익, 지수수익) 쌍을 전 종목에서 풀링한다.
# 종목 하나의 독립 표본은 252일 기준 11회가 상한이지만, 종목을 합치면
# 서로 다른 시기·종목의 창이 섞여 훨씬 많은 증거가 모인다.
pool = {h: {"s": [], "b": []} for h in HOLDS}
n_tickers = 0

for t, g in px.groupby("ticker"):
    g = g[["date", "close"]].sort_values("date")
    m = g.merge(bench, on="date", how="inner").reset_index(drop=True)
    if len(m) < 300:
        continue
    n_tickers += 1
    closes = m["close"].reset_index(drop=True)
    c = closes.to_numpy(dtype=float)
    b = m["bench"].to_numpy(dtype=float)
    scan = signals.ma_signal_scan(closes)
    ev = np.unique(np.concatenate(
        [np.asarray(scan[k]["events"], dtype=int) for k in KEYS if len(scan[k]["events"])]
    )) if any(len(scan[k]["events"]) for k in KEYS) else np.array([], dtype=int)
    if not len(ev):
        continue
    for h in HOLDS:
        v = ev[ev + h < len(c)]
        if not len(v):
            continue
        pool[h]["s"].append(c[v + h] / c[v] - 1.0)
        pool[h]["b"].append(b[v + h] / b[v] - 1.0)

print(f"대상 종목 {n_tickers}개 · 신호 {'+'.join(KEYS)} · KOSPI(^KS11) 대비\n")
print(f"{'보유':>6} {'표본(풀링)':>11} {'원시승률':>9} {'초과승률':>9} "
      f"{'평균수익':>9} {'지수수익':>9} {'초과수익':>9}")
print("-" * 70)
rows = []
for h in HOLDS:
    s = np.concatenate(pool[h]["s"])
    bb = np.concatenate(pool[h]["b"])
    raw_win = (s > 0).mean() * 100
    exc_win = (s > bb).mean() * 100
    rows.append((h, len(s), raw_win, exc_win, s.mean() * 100, bb.mean() * 100,
                 (s - bb).mean() * 100))
    print(f"{h:>4}일 {len(s):>11,} {raw_win:>8.1f}% {exc_win:>8.1f}% "
          f"{s.mean()*100:>8.1f}% {bb.mean()*100:>8.1f}% {(s-bb).mean()*100:>8.1f}%")

print()
first, last = rows[0], rows[-1]
print(f"2일 → 252일 변화:")
print(f"  원시 승률 : {first[2]:.1f}% → {last[2]:.1f}%  ({last[2]-first[2]:+.1f}%p)")
print(f"  초과 승률 : {first[3]:.1f}% → {last[3]:.1f}%  ({last[3]-first[3]:+.1f}%p)")
print(f"  지수 수익 : {first[5]:.1f}% → {last[5]:.1f}%  ({last[5]-first[5]:+.1f}%p)")
