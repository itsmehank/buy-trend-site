"""사이트의 실제 선정 기준이 KOSPI를 이겼는지 walk-forward로 측정.

포함 조건 3개(별점은 제외 기능이 없어 정렬만 바꾼다):
  1) RS >= 90
  2) 최적 보유기간이 잡히는 신호 존재 (표본>=20 & 손실 표본 존재)
  3) 현재가가 매수 구간(zone) 안

핵심 원칙: 각 as-of 시점 t에서 모든 입력을 t까지 자른다. 특히 최적 보유기간
선정도 t 시점 데이터로만 해야 한다(프로덕션은 전체 이력을 쓰므로, 그대로 쓰면
미래를 보고 기간을 고른 셈이 된다).

한계: 캐시에 상장폐지 종목이 없어 생존 편향은 제거되지 않는다. 대조군 B가 같은
      종목 풀에서 추출되므로 A-B 비교에서는 상쇄된다. 거래비용 미반영. KR 전용.
"""
import numpy as np
import pandas as pd

from pipeline import build, config
from pipeline.rs import rs_percentiles, rs_raw
from pipeline.zone import in_zone

STEP = 20                 # as-of 간격 (거래일) = 월 1회
MIN_BARS = 500            # 신호·백테스트에 필요한 최소 이력
HORIZONS = [20, 60, 126, 252]
TOP_N = 20                # EV 상위 N 부분집합도 함께 측정
rng = np.random.default_rng(20260726)

# ── 데이터: 벤치마크 거래일을 마스터 달력으로 삼아 wide 행렬 구성
px = pd.read_parquet("cache/prices_kr.parquet")
bench = (pd.read_parquet("cache/prices_bench.parquet")
         .query("ticker == '^KS11'")[["date", "close"]].sort_values("date"))
cal = bench["date"].to_numpy()
bench_px = bench["close"].to_numpy(dtype=float)

wide = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
             .reindex(cal) for c in ("close", "high", "low")}
tickers = list(wide["close"].columns)
C = wide["close"].to_numpy(dtype=float)
H = wide["high"].to_numpy(dtype=float)
L = wide["low"].to_numpy(dtype=float)
print(f"마스터 달력 {len(cal)}일 ({cal[0]} ~ {cal[-1]}) · 종목 {len(tickers)}개")

# ── as-of 시점: 이력이 쌓인 뒤부터, 최장 수평선이 들어가는 지점까지
starts = [np.argmax(~np.isnan(C[:, j])) for j in range(len(tickers))]
lo = int(np.percentile(starts, 20)) + MIN_BARS
hi = len(cal) - max(HORIZONS) - 1
asof_idx = list(range(lo, hi, STEP))
print(f"as-of 시점 {len(asof_idx)}개 ({cal[asof_idx[0]]} ~ {cal[asof_idx[-1]]})\n")

# 결과 누적: group -> horizon -> list of (종목초과수익, 이겼는지)
groups = ["A_사이트기준", f"A2_EV상위{TOP_N}", "B_무작위동수", "C_전체"]
acc = {g: {h: {"exc": [], "win": []} for h in HORIZONS} for g in groups}
per_date = {g: {h: [] for h in HORIZONS} for g in groups}   # 시점별 평균(독립 단위)
sel_sizes, elig_sizes = [], []

for i in asof_idx:
    # ── 적격 유니버스: 그 시점까지 MIN_BARS 이상 데이터가 있는 종목
    elig, raws = [], {}
    for j, t in enumerate(tickers):
        col = C[: i + 1, j]
        hist = col[~np.isnan(col)]
        if len(hist) < MIN_BARS or np.isnan(C[i, j]):
            continue
        r = rs_raw(hist)
        if r is not None:
            elig.append(j)
            raws[t] = r
    if len(elig) < 50:
        continue
    pct = rs_percentiles(raws)

    # ── 사이트 3개 필터 (전부 t까지 자른 데이터로)
    selected = []
    for j in elig:
        t = tickers[j]
        rs = pct.get(t)
        if rs is None or rs < config.RS_MIN:
            continue
        m = ~np.isnan(C[: i + 1, j])
        df = pd.DataFrame({"close": C[: i + 1, j][m],
                           "high": H[: i + 1, j][m],
                           "low": L[: i + 1, j][m]}).reset_index(drop=True)
        cands = build.ticker_signals(df)          # 여기서 최적기간도 t 기준으로 잡힌다
        if not cands:
            continue
        rep = max(cands, key=lambda c: c["best"]["ev"])
        if not in_zone(rep["signal"], rep["ref_price"], float(C[i, j])):
            continue
        selected.append((j, rep["best"]["ev"]))

    if not selected:
        continue
    sel_j = [j for j, _ in selected]
    top_j = [j for j, _ in sorted(selected, key=lambda x: -x[1])[:TOP_N]]
    rand_j = list(rng.choice(elig, size=min(len(sel_j), len(elig)), replace=False))
    sel_sizes.append(len(sel_j)); elig_sizes.append(len(elig))

    for gname, js in (("A_사이트기준", sel_j), (f"A2_EV상위{TOP_N}", top_j),
                      ("B_무작위동수", rand_j), ("C_전체", elig)):
        for h in HORIZONS:
            bret = bench_px[i + h] / bench_px[i] - 1.0
            ex = []
            for j in js:
                p0, p1 = C[i, j], C[i + h, j]
                if np.isnan(p0) or np.isnan(p1):
                    continue
                ex.append((p1 / p0 - 1.0) - bret)
            if ex:
                acc[gname][h]["exc"] += ex
                acc[gname][h]["win"] += [x > 0 for x in ex]
                per_date[gname][h].append(float(np.mean(ex)))

print(f"유효 시점 {len(sel_sizes)}개 · 시점당 선정 {np.mean(sel_sizes):.0f}종목 "
      f"(적격 {np.mean(elig_sizes):.0f}종목 중)\n")
print("=" * 78)
print("지수(KOSPI) 대비 초과수익 — 양수면 지수를 이김")
print("=" * 78)
for h in HORIZONS:
    print(f"\n[{h}일 보유]")
    print(f"  {'그룹':<14} {'평균초과':>9} {'중위초과':>9} {'지수이긴비율':>11} {'관측':>9}")
    print("  " + "-" * 60)
    for g in groups:
        e = np.array(acc[g][h]["exc"]) * 100
        w = np.mean(acc[g][h]["win"]) * 100
        print(f"  {g:<14} {e.mean():>8.2f}% {np.median(e):>8.2f}% "
              f"{w:>10.1f}% {len(e):>9,}")

print("\n" + "=" * 78)
print("A(사이트) - B(무작위) 차이 — 선정 기준의 순수 기여분")
print("=" * 78)
print(f"  {'보유':<8} {'평균초과 차이':>14} {'시점별 평균 차이':>18} {'A가 B를 이긴 시점':>18}")
print("  " + "-" * 62)
for h in HORIZONS:
    a = np.array(acc["A_사이트기준"][h]["exc"]).mean() * 100
    b = np.array(acc["B_무작위동수"][h]["exc"]).mean() * 100
    pa, pb = np.array(per_date["A_사이트기준"][h]), np.array(per_date["B_무작위동수"][h])
    n = min(len(pa), len(pb))
    wins = (pa[:n] > pb[:n]).sum()
    print(f"  {h:>4}일   {a - b:>13.2f}%p {(pa[:n] - pb[:n]).mean()*100:>17.2f}%p "
          f"{wins:>10}/{n} ({wins/n*100:.0f}%)")
