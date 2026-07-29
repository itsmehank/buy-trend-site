"""가설 진단 — 'n일 보유' 방식에서 손실 거래가 어떻게 진행됐는지 본다.
   손실 거래가 '중간에 추세가 깨진 뒤 계속 끌려간 것'이라면 이탈 매도가 살렸을 것이다."""
import numpy as np, pandas as pd
from pipeline import signals

px = pd.read_parquet("cache/prices_kr.parquet")
H = 252
rows = []
for t, g in px.groupby("ticker"):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 900: continue
    c = g["close"].to_numpy(float)
    sc = signals.ma_signal_scan(g["close"].reset_index(drop=True))
    ev = sc.get("L|SMA|20", {}).get("events", [])
    ev = np.asarray(ev, dtype=int)
    ev = ev[ev + H < len(c)]
    if not len(ev): continue
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    for i in ev:
        path = c[i:i+H+1]
        final = path[-1]/path[0] - 1
        peak = path.max()/path[0] - 1              # 보유 중 최고 미실현 수익
        trough = path.min()/path[0] - 1            # 보유 중 최저
        # 추세 이탈 시점: 종가가 SMA20 아래로 처음 마감한 날
        below = np.where(c[i+1:i+H+1] < sma20[i+1:i+H+1])[0]
        brk = int(below[0]) + 1 if len(below) else None
        rows.append((final, peak, trough, brk,
                     (c[i+brk]/c[i]-1) if brk else final))

df = pd.DataFrame(rows, columns=["final","peak","trough","brk_day","ret_at_break"])
print(f"252일 보유 거래 {len(df):,}건 (KR, L|SMA|20)\n")

L = df[df.final <= 0]
W = df[df.final > 0]
print(f"① 손실 거래 {len(L):,}건 ({len(L)/len(df)*100:.0f}%) 의 진행 양상")
print(f"   보유 중 한때 플러스였던 비율 : {(L.peak > 0).mean()*100:.0f}%")
print(f"   그때 최고 미실현 수익(중위)   : {L.peak.median()*100:+.1f}%")
print(f"   최종 수익(중위)              : {L.final.median()*100:+.1f}%")
print(f"   -> 이익을 반납하고 손실로 끝난 거래가 대부분인가?")

print(f"\n② 추세 이탈(종가 < SMA20) 시점")
print(f"   이탈 발생 비율      : {df.brk_day.notna().mean()*100:.0f}%")
print(f"   이탈까지 걸린 일수  : 중위 {df.brk_day.median():.0f}일 (252일 중)")

print(f"\n③ 만약 '이탈 즉시 매도'였다면 (같은 진입, 매도만 교체)")
for name, sub in (("전체", df), ("최종 손실 거래", L), ("최종 이익 거래", W)):
    print(f"   {name:<14} n일보유 중위 {sub.final.median()*100:>7.1f}%  "
          f"→ 이탈매도 중위 {sub.ret_at_break.median()*100:>7.1f}%")
print(f"\n   전체 평균 : {df.final.mean()*100:.1f}%  →  {df.ret_at_break.mean()*100:.1f}%")
