"""결과 검토 — 표본 끝의 KOSPI 폭등 국면(2025~2026)이 결론을 지배하는지 확인.
as-of 시점을 국면별로 나눠 A(사이트) vs B(무작위)를 다시 비교한다."""
import numpy as np, pandas as pd
from pipeline import build, config
from pipeline.rs import rs_percentiles, rs_raw
from pipeline.zone import in_zone

STEP, MIN_BARS = 20, 500
HS = [60, 252]
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

recs = []   # (asof_date, horizon, groupA평균, groupB평균)
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
        cd = build.ticker_signals(pd.DataFrame(
            {"close":C[:i+1,j][m],"high":Hi[:i+1,j][m],"low":Lo[:i+1,j][m]}).reset_index(drop=True))
        if not cd: continue
        rep = max(cd, key=lambda c: c["best"]["ev"])
        if in_zone(rep["signal"], rep["ref_price"], float(C[i,j])): sel.append(j)
    if not sel: continue
    rnd = list(rng.choice(elig, size=min(len(sel),len(elig)), replace=False))
    for h in HS:
        br = bpx[i+h]/bpx[i]-1
        out = {}
        for name, js in (("A",sel),("B",rnd)):
            e = [(C[i+h,j]/C[i,j]-1)-br for j in js
                 if not (np.isnan(C[i,j]) or np.isnan(C[i+h,j]))]
            out[name] = np.median(e) if e else np.nan     # 중위 = 이상치에 강건
            out[name+"m"] = np.mean(e) if e else np.nan
        recs.append((pd.Timestamp(cal[i]), h, out["Am"], out["Bm"], out["A"], out["B"]))

df = pd.DataFrame(recs, columns=["asof","h","A_mean","B_mean","A_med","B_med"])
df["국면"] = np.where(df["asof"] < "2024-01-01", "2016~2023 (정상장)", "2024~ (KOSPI 폭등)")

for h in HS:
    d = df[df.h == h]
    print(f"\n{'='*72}\n[{h}일 보유] 국면별 A(사이트) vs B(무작위) — 시점별 평균을 다시 평균\n{'='*72}")
    print(f"  {'국면':<20} {'시점':>5} {'A평균':>8} {'B평균':>8} {'차이':>8} {'A>B 시점':>10}")
    print("  " + "-"*64)
    for g, sub in d.groupby("국면", sort=True):
        am, bm = sub.A_mean.mean()*100, sub.B_mean.mean()*100
        wins = (sub.A_mean > sub.B_mean).sum()
        print(f"  {g:<20} {len(sub):>5} {am:>7.1f}% {bm:>7.1f}% {am-bm:>7.1f}%p "
              f"{wins:>6}/{len(sub)}")
    print(f"\n  중위값 기준 (이상치 영향 제거):")
    print(f"  {'국면':<20} {'A중위':>8} {'B중위':>8} {'차이':>8}")
    print("  " + "-"*48)
    for g, sub in d.groupby("국면", sort=True):
        am, bm = sub.A_med.mean()*100, sub.B_med.mean()*100
        print(f"  {g:<20} {am:>7.1f}% {bm:>7.1f}% {am-bm:>7.1f}%p")
