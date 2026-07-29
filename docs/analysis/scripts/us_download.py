"""US 유니버스 + 15년치 가격 다운로드 (cache/prices_us_test.parquet).

문서: docs/analysis/2026-07-29-strategy-validation.md
다운로드 성공률을 반드시 출력한다 — 조용히 빠진 티커가 있으면 생존편향이 커진다.
"""
import logging

from pipeline import data, universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

u = universe.us_stock_universe()
universe.save_universe("us_stock_test", u)
tks = u["ticker"].tolist()
print(f"NASDAQ 스크리너 통과 {len(tks):,}종목 — 15년치 다운로드 시작", flush=True)

df = data.fetch_us(tks, market="us_test", asof=None)
got = set(df["ticker"].unique())
miss = [t for t in tks if t not in got]
bars = df.groupby("ticker").size()

print("\n=== 다운로드 결과 ===")
print(f"성공 {len(got):,} / 요청 {len(tks):,} = {len(got)/len(tks)*100:.1f}%")
print(f"결측 {len(miss)}종목: {miss[:20]}")
print(f"봉 수: 중위 {bars.median():.0f} · 최소 {bars.min()} · 최대 {bars.max()}")
print(f"봉 500 미만(이력 부족, MIN_BARS에서 자동 제외): {(bars < 500).sum()}종목")
print(f"기간: {df['date'].min()} ~ {df['date'].max()}")
