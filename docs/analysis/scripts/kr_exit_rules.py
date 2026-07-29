"""진입 6종 x 매도 6종 전수 비교. 사전 판정 기준에 따라 전 칸을 공개한다.
한계: 거래비용 0.28% 가정, 생존편향 잔존, 재진입 없음(단일 거래), KR 전용."""
import numpy as np, pandas as pd
from pipeline import config, signals

MAXH, COST = 504, 0.28
EXITS = ["A_252일고정", "B_20일고정", "C_트레일링-15%", "D_SMA60하회",
         "E_SMA20하회5일", "F_트레일링-25%"]
ENTRIES = ["이평 L|SMA|20", "박스 L(60일)", "박스 S(20일)",
           "신고가 ATH", "신고가 20일", "신고가 55일"]

px = pd.read_parquet("cache/prices_kr.parquet")
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker=='^KS11'")
         [["date","close"]].sort_values("date"))
bmap = dict(zip(bench["date"], bench["close"].astype(float)))

acc = {e: {x: {"exc": [], "days": [], "fired": []} for x in EXITS} for e in ENTRIES}
brk_days = {e: [] for e in ENTRIES}

for t, g in px.groupby("ticker"):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 900: continue
    cs = g["close"].reset_index(drop=True)
    c = cs.to_numpy(float)
    bp = np.array([bmap.get(d, np.nan) for d in g["date"].to_numpy()])
    s20 = cs.rolling(20).mean().to_numpy()
    s60 = cs.rolling(60).mean().to_numpy()

    ma = signals.ma_signal_scan(cs)
    bx = signals.box_signal_scan(g["high"].reset_index(drop=True),
                                 g["low"].reset_index(drop=True), cs)
    nh = signals.nhigh_signal_scan(cs)
    ev_map = {
        "이평 L|SMA|20": ma.get("L|SMA|20", {}).get("events", []),
        "박스 L(60일)": bx["L"]["events"], "박스 S(20일)": bx["S"]["events"],
        "신고가 ATH": nh["ATH"]["events"], "신고가 20일": nh["20"]["events"],
        "신고가 55일": nh["55"]["events"],
    }

    for ename, evs in ev_map.items():
        evs = np.asarray(evs, dtype=int)
        evs = evs[(evs + MAXH < len(c))]
        for i in evs:
            if np.isnan(bp[i]): continue
            seg = c[i:i+MAXH+1]
            rmax = np.maximum.accumulate(seg)
            def first(mask):
                w = np.where(mask[1:])[0]
                return (int(w[0]) + 1, True) if len(w) else (MAXH, False)
            b20 = np.array([c[i+k] < s20[i+k] if not np.isnan(s20[i+k]) else False
                            for k in range(MAXH+1)])
            b60 = np.array([c[i+k] < s60[i+k] if not np.isnan(s60[i+k]) else False
                            for k in range(MAXH+1)])
            run, e5, f5 = 0, MAXH, False
            for k in range(1, MAXH+1):
                run = run + 1 if b20[k] else 0
                if run >= 5: e5, f5 = k, True; break
            d60, fd = first(b60)
            brk_days[ename].append(d60)
            plan = {"A_252일고정": (252, True), "B_20일고정": (20, True),
                    "C_트레일링-15%": first(seg <= rmax*0.85), "D_SMA60하회": (d60, fd),
                    "E_SMA20하회5일": (e5, f5), "F_트레일링-25%": first(seg <= rmax*0.75)}
            for x, (h, fired) in plan.items():
                if np.isnan(bp[i+h]): continue
                a = acc[ename][x]
                a["exc"].append((c[i+h]/c[i]-1) - (bp[i+h]/bp[i]-1))
                a["days"].append(h); a["fired"].append(fired)

print("진입 6종 x 매도 6종 — KOSPI 대비 초과수익 (KR, 거래비용 0.28% 반영)\n")
for e in ENTRIES:
    n = len(acc[e]["A_252일고정"]["exc"])
    bd = np.array(brk_days[e])
    print(f"\n{'='*82}\n■ {e}   거래 {n:,}건 · SMA60 이탈까지 중위 {np.median(bd):.0f}일\n{'='*82}")
    print(f"  {'매도':<16} {'평균초과':>8} {'중위초과':>8} {'승률':>7} {'보유':>7} "
          f"{'발동률':>7} {'비용후연율화':>11}")
    print("  " + "-"*72)
    for x in EXITS:
        a = acc[e][x]
        if not a["exc"]: continue
        ex = np.array(a["exc"])*100; dy = np.array(a["days"])
        tpy = 252/dy.mean()
        net = ex.mean()*tpy - tpy*COST
        print(f"  {x:<16} {ex.mean():>7.1f}% {np.median(ex):>7.1f}% "
              f"{(ex>0).mean()*100:>6.1f}% {dy.mean():>6.0f}일 "
              f"{np.mean(a['fired'])*100:>6.0f}% {net:>10.1f}%")
