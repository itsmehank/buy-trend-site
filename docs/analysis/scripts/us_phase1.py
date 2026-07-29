"""Phase 1 — US 기여 분해: RS 필터 vs 신호+zone 필터.
KR 테스트와 완전 동일 파라미터(STEP=20, MIN_BARS=500, RS_MIN=90).

사전 판정 기준: 신호 기여 인정 = (A-D)가 평균·중위 둘 다 양수, 여러 수평선에서 일관.
한계: 생존편향 잔존(스크리너는 현재 상장분만) -> 절대값 아닌 차분만 해석. ETF 제외.
"""
import numpy as np
import pandas as pd

from pipeline import build, config, data, universe
from pipeline.rs import rs_percentiles, rs_raw
from pipeline.zone import in_zone

STEP, MIN_BARS = 20, 500
HORIZONS = [20, 60, 126, 252]
rng = np.random.default_rng(20260729)

px = data.load_cache("us_test")
bench = (pd.read_parquet("cache/prices_bench.parquet").query("ticker == '^GSPC'")
         [["date", "close"]].sort_values("date"))
cal, bpx = bench["date"].to_numpy(), bench["close"].to_numpy(float)

# 거래대금 필터는 as-of 시점마다 다시 계산한다. universe.dollar_vol_filter는
# df.tail(20) = '오늘' 기준이라 과거 시점에 적용하면 look-ahead가 된다.
W = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
         .reindex(cal) for c in ("close", "high", "low", "volume")}
tk = list(W["close"].columns)
C, Hi, Lo, V = (W[c].to_numpy(float) for c in ("close", "high", "low", "volume"))
# 시점 i까지의 직전 20봉 평균 거래대금 (as-of 유동성)
DV = pd.DataFrame(C * V).rolling(20, min_periods=20).mean().to_numpy()
print(f"유니버스 {len(tk):,}종목 (거래대금 필터는 시점별로 적용)")
starts = [np.argmax(~np.isnan(C[:, j])) for j in range(len(tk))]
idxs = list(range(int(np.percentile(starts, 20)) + MIN_BARS, len(cal) - max(HORIZONS) - 1, STEP))
print(f"마스터 달력 {len(cal)}일 · as-of {len(idxs)}시점 ({cal[idxs[0]]} ~ {cal[idxs[-1]]})\n")

GROUPS = ["A_사이트", "B_전체무작위", "D_RS90내무작위", "E_RS90전체"]
res = {g: {h: [] for h in HORIZONS} for g in GROUPS}
sizes = {"적격": [], "RS>=90": [], "신호": [], "최종": []}

for i in idxs:
    elig, raws = [], {}
    for j, t in enumerate(tk):
        col = C[:i + 1, j]
        h = col[~np.isnan(col)]
        if len(h) < MIN_BARS or np.isnan(C[i, j]):
            continue
        # as-of 유동성: 그 시점 직전 20봉 평균 거래대금 >= $2M
        if not (DV[i, j] >= config.US_MIN_DOLLAR_VOL):
            continue
        r = rs_raw(h)
        # 종가 0인 종목(CBIO/DEC)이 있어 rs_raw가 NaN/inf를 낼 수 있다.
        # rs_percentiles는 NaN을 int로 못 바꿔 죽으므로 여기서 제외한다.
        if r is not None and np.isfinite(r):
            elig.append(j); raws[t] = r
    if len(elig) < 100:
        continue
    pct = rs_percentiles(raws)
    rs90 = [j for j in elig if (pct.get(tk[j]) or 0) >= config.RS_MIN]
    nsig, sel = 0, []
    for j in rs90:
        m = ~np.isnan(C[:i + 1, j])
        cd = build.ticker_signals(pd.DataFrame(
            {"close": C[:i + 1, j][m], "high": Hi[:i + 1, j][m],
             "low": Lo[:i + 1, j][m]}).reset_index(drop=True))
        if not cd:
            continue
        nsig += 1
        rep = max(cd, key=lambda c: c["best"]["ev"])
        if in_zone(rep["signal"], rep["ref_price"], float(C[i, j])):
            sel.append(j)
    if not sel:
        continue
    for k, v in zip(sizes, (len(elig), len(rs90), nsig, len(sel))):
        sizes[k].append(v)

    grp = {"A_사이트": sel,
           "B_전체무작위": list(rng.choice(elig, size=min(len(sel), len(elig)), replace=False)),
           "D_RS90내무작위": list(rng.choice(rs90, size=min(len(sel), len(rs90)), replace=False)),
           "E_RS90전체": rs90}
    for h in HORIZONS:
        br = bpx[i + h] / bpx[i] - 1.0
        for g, js in grp.items():
            e = [(C[i + h, j] / C[i, j] - 1.0) - br for j in js
                 if not (np.isnan(C[i, j]) or np.isnan(C[i + h, j]))]
            if e:
                res[g][h].append((float(np.mean(e)), float(np.median(e))))

print("단계별 통과 (시점 평균)")
prev = None
for k, v in sizes.items():
    m = np.mean(v)
    print(f"  {k:<10} {m:>7.1f}종목" + (f"  (직전 대비 {m/prev*100:.0f}%)" if prev else ""))
    prev = m

print(f"\n{'='*74}\nS&P500 대비 초과수익 — 시점별 값을 다시 평균 ({len(sizes['최종'])}시점)\n{'='*74}")
for h in HORIZONS:
    print(f"\n[{h}일 보유]")
    print(f"  {'그룹':<16} {'평균초과':>9} {'중위초과':>9}")
    print("  " + "-" * 38)
    for g in GROUPS:
        a = np.array(res[g][h])
        print(f"  {g:<16} {a[:,0].mean()*100:>8.1f}% {a[:,1].mean()*100:>8.1f}%")
    A, B, D = (np.array(res[g][h]) for g in ("A_사이트", "B_전체무작위", "D_RS90내무작위"))
    print(f"  → RS 필터 기여   (D−B): 평균 {(D[:,0].mean()-B[:,0].mean())*100:>6.1f}%p "
          f"· 중위 {(D[:,1].mean()-B[:,1].mean())*100:>6.1f}%p")
    print(f"  → 신호+zone 기여 (A−D): 평균 {(A[:,0].mean()-D[:,0].mean())*100:>6.1f}%p "
          f"· 중위 {(A[:,1].mean()-D[:,1].mean())*100:>6.1f}%p")
