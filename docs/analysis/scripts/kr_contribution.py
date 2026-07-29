"""검증 — (1) RS>=90 필터가 실제로 몇 종목을 통과시키는지 단계별 확인
        (2) 대조군을 'RS>=90 안에서 무작위'로 바꿔 신호+zone의 순수 기여 분리"""
import numpy as np, pandas as pd
from pipeline import build, config
from pipeline.rs import rs_percentiles, rs_raw
from pipeline.zone import in_zone

STEP, MIN_BARS, HS = 20, 500, [60, 252]
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
idxs = list(range(int(np.percentile(starts,20))+MIN_BARS, len(cal)-max(HS)-1, STEP))

fun = {"적격": [], "RS>=90": [], "신호있음": [], "zone내(최종)": []}
res = {g: {h: [] for h in HS} for g in
       ("A_사이트", "B_전체무작위", "D_RS90내무작위", "E_RS90전체")}

for i in idxs:
    elig, raws = [], {}
    for j,t in enumerate(tk):
        col = C[:i+1,j]; h = col[~np.isnan(col)]
        if len(h) < MIN_BARS or np.isnan(C[i,j]): continue
        r = rs_raw(h)
        if r is not None: elig.append(j); raws[t]=r
    if len(elig) < 50: continue
    pct = rs_percentiles(raws)
    rs90 = [j for j in elig if (pct.get(tk[j]) or 0) >= config.RS_MIN]
    has_sig, sel = [], []
    for j in rs90:
        m = ~np.isnan(C[:i+1,j])
        cd = build.ticker_signals(pd.DataFrame(
            {"close":C[:i+1,j][m],"high":Hi[:i+1,j][m],"low":Lo[:i+1,j][m]}).reset_index(drop=True))
        if not cd: continue
        has_sig.append(j)
        rep = max(cd, key=lambda c: c["best"]["ev"])
        if in_zone(rep["signal"], rep["ref_price"], float(C[i,j])): sel.append(j)
    if not sel: continue
    for k, v in zip(fun, (len(elig), len(rs90), len(has_sig), len(sel))): fun[k].append(v)

    grp = {
        "A_사이트": sel,
        "B_전체무작위": list(rng.choice(elig, size=min(len(sel), len(elig)), replace=False)),
        "D_RS90내무작위": list(rng.choice(rs90, size=min(len(sel), len(rs90)), replace=False)),
        "E_RS90전체": rs90,
    }
    for h in HS:
        br = bpx[i+h]/bpx[i]-1
        for g, js in grp.items():
            e = [(C[i+h,j]/C[i,j]-1)-br for j in js
                 if not (np.isnan(C[i,j]) or np.isnan(C[i+h,j]))]
            if e: res[g][h].append((np.mean(e), np.median(e)))

print("① 단계별 통과 종목 수 (시점 평균)")
prev = None
for k, v in fun.items():
    m = np.mean(v)
    rate = f"  (직전 대비 {m/prev*100:.0f}%)" if prev else ""
    print(f"   {k:<14} {m:>6.1f}종목{rate}")
    prev = m

print(f"\n② 대조군 비교 — RS 필터의 기여 vs 신호+zone의 기여  ({len(fun['적격'])}시점)")
for h in HS:
    print(f"\n  [{h}일 보유]  시점별 값을 다시 평균")
    print(f"    {'그룹':<16} {'평균초과':>9} {'중위초과':>9}")
    print("    " + "-"*36)
    for g in ("B_전체무작위", "E_RS90전체", "D_RS90내무작위", "A_사이트"):
        arr = np.array(res[g][h])
        print(f"    {g:<16} {arr[:,0].mean()*100:>8.1f}% {arr[:,1].mean()*100:>8.1f}%")
    a = np.array(res["A_사이트"][h]); d = np.array(res["D_RS90내무작위"][h])
    b = np.array(res["B_전체무작위"][h]); e = np.array(res["E_RS90전체"][h])
    print(f"    → RS 필터 기여   (E−B): 평균 {(e[:,0].mean()-b[:,0].mean())*100:>6.1f}%p "
          f"· 중위 {(e[:,1].mean()-b[:,1].mean())*100:>6.1f}%p")
    print(f"    → 신호+zone 기여 (A−D): 평균 {(a[:,0].mean()-d[:,0].mean())*100:>6.1f}%p "
          f"· 중위 {(a[:,1].mean()-d[:,1].mean())*100:>6.1f}%p")
