"""결과 검토 — 평균 초과수익이 소수 이상치에 끌려간 것인지, 그리고 시점별로
얼마나 일관적인지 확인한다. (walk-forward 스크립트와 동일 로직, 분포만 추가 출력)"""
import numpy as np, pandas as pd
from pipeline import build, config
from pipeline.rs import rs_percentiles, rs_raw
from pipeline.zone import in_zone

STEP, MIN_BARS, H = 20, 500, 252
rng = np.random.default_rng(20260726)
px = pd.read_parquet("cache/prices_kr.parquet")
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker=='^KS11'")
         [["date","close"]].sort_values("date"))
cal, bpx = bench["date"].to_numpy(), bench["close"].to_numpy(float)
W = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last").reindex(cal)
     for c in ("close","high","low")}
tk = list(W["close"].columns)
C, Hi, Lo = (W[c].to_numpy(float) for c in ("close","high","low"))
starts = [np.argmax(~np.isnan(C[:,j])) for j in range(len(tk))]
idxs = list(range(int(np.percentile(starts,20))+MIN_BARS, len(cal)-H-1, STEP))

A, B, dateA, dateB = [], [], [], []
for i in idxs:
    elig, raws = [], {}
    for j,t in enumerate(tk):
        col = C[:i+1,j]; h = col[~np.isnan(col)]
        if len(h) < MIN_BARS or np.isnan(C[i,j]): continue
        r = rs_raw(h)
        if r is not None: elig.append(j); raws[t]=r
    if len(elig) < 50: continue
    pct = rs_percentiles(raws)
    sel = []
    for j in elig:
        if (pct.get(tk[j]) or 0) < config.RS_MIN: continue
        m = ~np.isnan(C[:i+1,j])
        df = pd.DataFrame({"close":C[:i+1,j][m],"high":Hi[:i+1,j][m],"low":Lo[:i+1,j][m]})
        cd = build.ticker_signals(df.reset_index(drop=True))
        if not cd: continue
        rep = max(cd, key=lambda c: c["best"]["ev"])
        if in_zone(rep["signal"], rep["ref_price"], float(C[i,j])): sel.append(j)
    if not sel: continue
    rnd = list(rng.choice(elig, size=min(len(sel),len(elig)), replace=False))
    br = bpx[i+H]/bpx[i]-1
    for js, out, dout in ((sel,A,dateA),(rnd,B,dateB)):
        e = [(C[i+H,j]/C[i,j]-1)-br for j in js
             if not (np.isnan(C[i,j]) or np.isnan(C[i+H,j]))]
        out += e
        if e: dout.append(np.mean(e))

a, b = np.array(A)*100, np.array(B)*100
print(f"252일 보유 · A(사이트) {len(a):,}관측 / B(무작위) {len(b):,}관측\n")
print("① 평균이 이상치에 끌려갔나 — 상위 관측을 잘라내며 평균 재계산")
print(f"  {'제외':<16} {'A 평균':>9} {'B 평균':>9} {'A-B':>9}")
print("  " + "-"*46)
for k in (0, 1, 5, 10):
    aa = np.sort(a)[:len(a)-int(len(a)*k/100)] if k else a
    bb = np.sort(b)[:len(b)-int(len(b)*k/100)] if k else b
    print(f"  {'상위 '+str(k)+'% 제외':<16} {aa.mean():>8.2f}% {bb.mean():>8.2f}% {aa.mean()-bb.mean():>8.2f}%p")
print(f"\n  절사평균(상하 10%씩): A {np.mean(np.sort(a)[len(a)//10:-len(a)//10]):.2f}%  "
      f"B {np.mean(np.sort(b)[len(b)//10:-len(b)//10]):.2f}%")

print("\n② 상위 소수가 전체 평균에 기여하는 비중")
for k in (1, 5):
    top = np.sort(a)[-int(len(a)*k/100):]
    print(f"  A 상위 {k}%({len(top)}개)가 전체 합의 {top.sum()/a.sum()*100:.0f}% 차지")

print("\n③ 분포")
print(f"  {'':<6} {'평균':>8} {'중위':>8} {'표준편차':>9} {'최대':>9} {'최소':>9}")
print(f"  {'A':<6} {a.mean():>7.1f}% {np.median(a):>7.1f}% {a.std():>8.1f}% {a.max():>8.0f}% {a.min():>8.0f}%")
print(f"  {'B':<6} {b.mean():>7.1f}% {np.median(b):>7.1f}% {b.std():>8.1f}% {b.max():>8.0f}% {b.min():>8.0f}%")

print("\n④ 독립 단위(겹치지 않는 252일 창)로 본 일관성")
pa, pb = np.array(dateA), np.array(dateB)
step = H // STEP                      # 252/20 ≈ 12시점마다 창이 겹치지 않는다
ia, ib = pa[::step], pb[::step]
print(f"  겹치지 않는 시점 {len(ia)}개 (전체 {len(pa)}개 중)")
print(f"  A가 B를 이긴 창: {(ia>ib).sum()}/{len(ia)}")
print(f"  A 평균 {ia.mean()*100:.1f}%  B 평균 {ib.mean()*100:.1f}%  차이 {(ia-ib).mean()*100:.1f}%p")
