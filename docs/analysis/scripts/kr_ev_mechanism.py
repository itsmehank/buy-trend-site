"""결정 2 최종 — 드리프트를 제거해도 252일이 이기는 잔여 원인 규명.

가설: 장기 수익 분포는 오른쪽으로 치우쳐(right-skew) 손익비가 커진다.
      상승은 복리로 무한히 열려 있고 하락은 -100%에 막혀 있기 때문.
      그렇다면 엣지가 0이어도 EV는 보유기간에 따라 오른다.
검증: 초과수익(드리프트 제거) 기준으로 승률과 손익비를 분리해 본다.
      추가로 '엣지 0' 대조군 — 신호와 무관한 무작위 날짜로 같은 계산.
"""
import numpy as np, pandas as pd
from pipeline import config, signals

HOLDS, KEYS = config.HOLDS_MA, ["L|SMA|20", "S|SMA|20"]
px = pd.read_parquet("cache/prices_kr.parquet")
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker == '^KS11'")
         [["date", "close"]].rename(columns={"close": "bench"}))

rng = np.random.default_rng(20260726)
sig = {h: [] for h in HOLDS}     # 신호 기준 초과수익
rnd = {h: [] for h in HOLDS}     # 무작위 날짜 초과수익 (엣지 0 대조군)

for t, g in px.groupby("ticker"):
    m = g[["date", "close"]].sort_values("date").merge(bench, on="date").reset_index(drop=True)
    if len(m) < 300:
        continue
    closes = m["close"].reset_index(drop=True)
    c, b = closes.to_numpy(float), m["bench"].to_numpy(float)
    scan = signals.ma_signal_scan(closes)
    evs = [np.asarray(scan[k]["events"], dtype=int) for k in KEYS if len(scan[k]["events"])]
    if not evs:
        continue
    ev = np.unique(np.concatenate(evs))
    fake = rng.integers(0, len(c), size=len(ev))     # 같은 개수의 무작위 날짜
    for h in HOLDS:
        for src, dst in ((ev, sig), (fake, rnd)):
            v = src[src + h < len(c)]
            if len(v):
                dst[h].append((c[v + h] / c[v]) - (b[v + h] / b[v]))

def stats(rets):
    r = np.concatenate(rets) * 100
    w, l = r[r > 0], r[r <= 0]
    p = len(w) / len(r) * 100
    aw, al = (w.mean() if len(w) else 0.0), l.mean()
    plr = aw / abs(al)
    return p, aw, al, plr, (p/100*aw + (1-p/100)*al) / abs(al)

print("초과수익(지수 차감) 기준 — 승률과 손익비를 분리\n")
print(f"{'보유':>6} | {'신호 기준':>34} | {'무작위 날짜(엣지 0 대조군)':>30}")
print(f"{'':>6} | {'승률':>6} {'평균이익':>8} {'평균손실':>8} {'손익비':>6} | {'승률':>6} {'손익비':>6} {'EV':>7}")
print("-" * 80)
for h in HOLDS:
    p, aw, al, plr, e = stats(sig[h])
    p2, _, _, plr2, e2 = stats(rnd[h])
    print(f"{h:>4}일 | {p:>5.1f}% {aw:>7.1f}% {al:>7.1f}% {plr:>6.2f} | {p2:>5.1f}% {plr2:>6.2f} {e2:>7.3f}")

print()
p2, aw2, al2, plr2, e2 = stats(rnd[2]); p252, _, _, plr252, e252 = stats(rnd[252])
print("대조군(무작위 날짜 = 신호 엣지 0, 드리프트도 제거됨) 결과:")
print(f"  손익비 : {plr2:.2f} (2일) → {plr252:.2f} (252일)")
print(f"  EV     : {e2:.3f} (2일) → {e252:.3f} (252일)")
print()
print("=> 엣지가 0이고 드리프트도 없는데 EV가 보유기간에 따라 오르면,")
print("   상승 원인은 신호 실력도 시장 상승도 아니라 '분포의 치우침'이다.")
