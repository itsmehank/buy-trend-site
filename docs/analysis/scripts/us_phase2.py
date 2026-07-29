"""Phase 2 — US 매도 규칙 6종 x 진입 6종. KR 테스트와 동일 규칙·파라미터.

거래비용은 US 0.10%(거래세 없음)와 KR 기준 0.28% 둘 다 표기해 민감도를 노출한다.
사전 판정: 매도 가설 지지 = D 또는 E가 A보다 비용후 연율화 ↑ 그리고 중위 개선, 여러 진입 일관.
한계: 생존편향 잔존, 재진입 없음(단일 거래), 연율화는 선형 환산(짧은 보유에 유리).
"""
import numpy as np
import pandas as pd

from pipeline import config, data, signals, universe

MAXH = 504
DAYS = np.arange(1, MAXH + 1)        # 이벤트 이후 경과일수 (E 규칙 상한용)
EXITS = ["A_252일고정", "B_20일고정", "C_트레일링-15%", "D_SMA60하회",
         "E_SMA20하회5일", "F_트레일링-25%"]
ENTRIES = ["이평 L|SMA|20", "박스 L(60일)", "박스 S(20일)",
           "신고가 ATH", "신고가 20일", "신고가 55일"]

px = data.load_cache("us_test")
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker == '^GSPC'")
         [["date", "close"]].sort_values("date"))
bmap = dict(zip(bench["date"], bench["close"].astype(float)))
print(f"유니버스 {px['ticker'].nunique():,}종목 "
      f"(거래대금 필터는 각 진입 시점 기준으로 적용)")

acc = {e: {x: {"exc": [], "days": []} for x in EXITS} for e in ENTRIES}
brk = {e: [] for e in ENTRIES}
done = 0

for t, g in px.groupby("ticker"):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 900:
        continue
    c = g["close"].to_numpy(float)
    if not np.all(c > 0):            # 종가 0 종목(CBIO/DEC) 제외
        continue
    done += 1
    bp = np.array([bmap.get(d, np.nan) for d in g["date"].to_numpy()])
    cs = g["close"].reset_index(drop=True)
    s20 = cs.rolling(20).mean().to_numpy()
    s60 = cs.rolling(60).mean().to_numpy()
    # 진입 시점 기준 유동성 (universe.dollar_vol_filter의 tail(20)은 '오늘'
    # 기준이라 과거 이벤트에 적용하면 look-ahead가 된다)
    dv = (cs * g["volume"]).rolling(20, min_periods=20).mean().to_numpy()

    b20 = np.where(np.isnan(s20), False, c < s20)
    b60 = np.where(np.isnan(s60), False, c < s60)
    idx60 = np.flatnonzero(b60)
    # E 규칙: 이벤트 '이후' 5일 연속 하회. 전역 rolling으로 구하면 이벤트 전에
    # 시작된 연속을 이어받아 1~4일에 조기 발동한다(루프 버전과 25~88건 불일치).
    # 각 시점에서 '그 시점부터 세는' 연속 카운트를 따로 만들어 해결한다.
    runlen = np.zeros(len(c), dtype=int)
    for k in range(1, len(c)):
        runlen[k] = runlen[k - 1] + 1 if b20[k] else 0

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
        evs = evs[evs + MAXH < len(c)]
        for i in evs:
            if np.isnan(bp[i]) or not (dv[i] >= config.US_MIN_DOLLAR_VOL):
                continue
            seg = c[i:i + MAXH + 1]
            rmax = np.maximum.accumulate(seg)

            def trail(k):
                w = np.flatnonzero(seg[1:] <= rmax[1:] * k)
                return int(w[0]) + 1 if len(w) else MAXH

            def after(idx):
                p = np.searchsorted(idx, i + 1, "left")
                return min(int(idx[p]) - i, MAXH) if p < len(idx) else MAXH

            # 이벤트 이후부터 새로 센 연속 5일.
            # 전역 runlen을 '이벤트 이후 경과일수'로 상한 처리하면, 이벤트 전에
            # 시작된 연속을 이어받지 않고 i+1부터 새로 센 것과 같아진다.
            seg_run = np.minimum(runlen[i + 1:i + MAXH + 1], DAYS)
            w5 = np.flatnonzero(seg_run >= 5)
            e5 = int(w5[0]) + 1 if len(w5) else MAXH

            d60 = after(idx60)
            brk[ename].append(d60)
            plan = {"A_252일고정": 252, "B_20일고정": 20, "C_트레일링-15%": trail(0.85),
                    "D_SMA60하회": d60, "E_SMA20하회5일": e5,
                    "F_트레일링-25%": trail(0.75)}
            for x, h in plan.items():
                if np.isnan(bp[i + h]):
                    continue
                a = acc[ename][x]
                a["exc"].append((c[i + h] / c[i] - 1) - (bp[i + h] / bp[i] - 1))
                a["days"].append(h)

print(f"처리 완료 {done:,}종목\n")
print("진입 6종 x 매도 6종 — S&P500 대비 초과수익 (US)")
for e in ENTRIES:
    n = len(acc[e]["A_252일고정"]["exc"])
    if not n:
        continue
    print(f"\n{'='*88}\n■ {e}   거래 {n:,}건 · SMA60 이탈까지 중위 {np.median(brk[e]):.0f}일\n{'='*88}")
    print(f"  {'매도':<16} {'평균초과':>8} {'중위초과':>8} {'승률':>7} {'보유':>7} "
          f"{'연율화@0.10%':>12} {'연율화@0.28%':>12}")
    print("  " + "-" * 78)
    for x in EXITS:
        a = acc[e][x]
        if not a["exc"]:
            continue
        ex = np.array(a["exc"]) * 100
        dy = np.array(a["days"])
        tpy = 252 / dy.mean()
        g10 = ex.mean() * tpy - tpy * 0.10
        g28 = ex.mean() * tpy - tpy * 0.28
        print(f"  {x:<16} {ex.mean():>7.1f}% {np.median(ex):>7.1f}% "
              f"{(ex>0).mean()*100:>6.1f}% {dy.mean():>6.0f}일 {g10:>11.1f}% {g28:>11.1f}%")
