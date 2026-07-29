"""btlib 회귀 검증 — 하네스가 2026-07-29 검증 결과를 재현하는지 대조한다.

하네스를 수정하면 반드시 다시 실행할 것. (PROTOCOL §4-(b))

기준값 출처:
  KR — 원본 docs/analysis/scripts/kr_exit_rules.py 실행 출력 (2026-07-29 재실행,
       데이터 기준일 2026-07-21). 원본은 §5 버그 수정 미적용이지만 KR은 해당 버그의
       영향을 받지 않으므로(문서 §10) 유효한 기준선이다.
  US — 문서 §6 Phase 2 표 (수정본 스크립트, 데이터 기준일 2026-07-28). 거래수는
       정수 일치, 연율화는 표기 반올림(±0.06) 일치를 요구한다.

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis python docs/analysis/btlib/regression_check.py kr
  PYTHONPATH=.:docs/analysis python docs/analysis/btlib/regression_check.py us
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # docs/analysis
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # 저장소 루트

import numpy as np  # noqa: E402

from btlib import engine, entries, loading, metrics  # noqa: E402
from pipeline import config  # noqa: E402

EXITS = {
    "A_252일고정": ("fixed", 252), "B_20일고정": ("fixed", 20),
    "C_트레일링-15%": ("trailing", 0.85), "D_SMA60하회": ("sma_below", 60),
    "E_SMA20하회5일": ("consec_below", (20, 5)), "F_트레일링-25%": ("trailing", 0.75),
}

# KR 기준값: {진입: (거래수, {매도: (평균, 중위, 승률, 보유, 연율화@0.28)})}
KR_BASE = {
    "이평 L|SMA|20": (48403, {
        "A_252일고정": (11.7, -2.3, 47.0, 252, 11.4), "B_20일고정": (0.7, -1.3, 44.4, 20, 5.9),
        "C_트레일링-15%": (2.5, -5.2, 37.7, 67, 8.2), "D_SMA60하회": (1.1, -1.6, 32.0, 18, 11.3),
        "E_SMA20하회5일": (1.0, -2.6, 31.1, 18, 10.3), "F_트레일링-25%": (4.7, -7.8, 37.5, 158, 7.0)}),
    "박스 L(60일)": (1410, {
        "A_252일고정": (5.1, -1.8, 48.2, 252, 4.8), "B_20일고정": (0.6, -0.2, 48.5, 20, 4.0),
        "C_트레일링-15%": (1.9, -3.0, 42.7, 130, 3.1), "D_SMA60하회": (0.5, -2.5, 36.8, 36, 1.2),
        "E_SMA20하회5일": (0.6, -1.6, 40.0, 28, 3.3), "F_트레일링-25%": (2.4, -5.1, 41.5, 261, 2.1)}),
    "박스 S(20일)": (11852, {
        "A_252일고정": (11.1, -2.7, 46.6, 252, 10.9), "B_20일고정": (0.8, -0.9, 45.9, 20, 6.0),
        "C_트레일링-15%": (3.1, -4.5, 38.4, 78, 9.1), "D_SMA60하회": (2.1, -2.0, 32.8, 27, 17.3),
        "E_SMA20하회5일": (1.5, -2.6, 37.6, 27, 11.4), "F_트레일링-25%": (5.2, -7.3, 37.8, 182, 6.8)}),
    "신고가 ATH": (11852, {
        "A_252일고정": (18.4, 0.5, 50.6, 252, 18.1), "B_20일고정": (2.8, 0.1, 50.3, 20, 32.2),
        "C_트레일링-15%": (5.6, -4.0, 41.6, 58, 23.1), "D_SMA60하회": (5.8, -3.2, 41.3, 47, 29.7),
        "E_SMA20하회5일": (4.3, -2.3, 42.1, 29, 34.7), "F_트레일링-25%": (6.9, -6.7, 40.5, 135, 12.3)}),
    "신고가 20일": (72370, {
        "A_252일고정": (13.2, -2.4, 47.1, 252, 12.9), "B_20일고정": (1.1, -0.8, 46.6, 20, 10.3),
        "C_트레일링-15%": (2.9, -4.7, 38.2, 74, 8.8), "D_SMA60하회": (2.3, -2.3, 33.4, 30, 16.5),
        "E_SMA20하회5일": (1.6, -2.8, 36.6, 26, 13.1), "F_트레일링-25%": (5.7, -7.3, 38.2, 170, 8.1)}),
    "신고가 55일": (39791, {
        "A_252일고정": (13.4, -2.0, 47.6, 252, 13.1), "B_20일고정": (1.3, -1.0, 46.3, 20, 12.4),
        "C_트레일링-15%": (2.9, -4.9, 38.1, 69, 9.7), "D_SMA60하회": (2.9, -4.0, 34.9, 39, 16.9),
        "E_SMA20하회5일": (1.9, -3.0, 36.9, 26, 15.3), "F_트레일링-25%": (5.5, -7.3, 38.2, 163, 8.1)}),
}

# US 기준값 (문서 §6 표): {진입: (거래수, A@0.10, A@0.28, D@0.10, D@0.28)}
US_BASE = {
    "이평 L|SMA|20": (291868, 1.0, 0.8, 1.9, -0.0),
    "박스 L(60일)": (16130, -0.9, -1.0, -0.6, -1.6),
    "박스 S(20일)": (60885, -0.6, -0.7, -2.5, -3.9),
    "신고가 ATH": (130162, 1.5, 1.3, 0.3, -0.7),
    "신고가 20일": (427524, 1.6, 1.4, 0.3, -1.0),
    "신고가 55일": (278990, 1.5, 1.3, 0.1, -1.0),
}

TOL = 0.06  # 표기 반올림(0.1 단위) 허용 오차


def check(label, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    if not ok:
        print(f"  FAIL {label}: got {got:.3f} want {want}")
    return ok


def run_kr():
    px, bench = loading.load_prices("kr"), loading.load_bench("kr")
    res = engine.run_event_backtest(px, bench, entries.STANDARD_ENTRIES, EXITS)
    fails = 0
    for e, (n_want, rows) in KR_BASE.items():
        n_got = len(res[e]["A_252일고정"]["exc"])
        if n_got != n_want:
            print(f"  FAIL {e} 거래수: got {n_got} want {n_want}")
            fails += 1
        for x, (mean, med, win, days, net28) in rows.items():
            s = metrics.summarize(res[e][x]["exc"], res[e][x]["days"], cost_pcts=(0.28,))
            fails += sum(not v for v in [
                check(f"{e}/{x} 평균", s["mean_pct"], mean),
                check(f"{e}/{x} 중위", s["median_pct"], med),
                check(f"{e}/{x} 승률", s["winrate_pct"], win),
                check(f"{e}/{x} 보유", s["avg_days"], days, tol=0.5),
                check(f"{e}/{x} 연율화", s["net_ann_pct"]["0.28"], net28),
            ])
    print(f"KR 회귀: {'PASS' if fails == 0 else f'FAIL {fails}건'} "
          f"(데이터 기준일 {loading.asof(px)})")
    return fails


def run_us():
    px, bench = loading.load_prices("us"), loading.load_bench("us")
    res = engine.run_event_backtest(px, bench, entries.STANDARD_ENTRIES, EXITS,
                                    min_dollar_vol=config.US_MIN_DOLLAR_VOL)
    fails = 0
    for e, (n_want, a10, a28, d10, d28) in US_BASE.items():
        n_got = len(res[e]["A_252일고정"]["exc"])
        if n_got != n_want:
            print(f"  FAIL {e} 거래수: got {n_got} want {n_want}")
            fails += 1
        sa = metrics.summarize(res[e]["A_252일고정"]["exc"], res[e]["A_252일고정"]["days"])
        sd = metrics.summarize(res[e]["D_SMA60하회"]["exc"], res[e]["D_SMA60하회"]["days"])
        fails += sum(not v for v in [
            check(f"{e}/A @0.10", sa["net_ann_pct"]["0.10"], a10),
            check(f"{e}/A @0.28", sa["net_ann_pct"]["0.28"], a28),
            check(f"{e}/D @0.10", sd["net_ann_pct"]["0.10"], d10),
            check(f"{e}/D @0.28", sd["net_ann_pct"]["0.28"], d28),
        ])
    print(f"US 회귀: {'PASS' if fails == 0 else f'FAIL {fails}건'} "
          f"(처리 {res['_n_tickers']:,}종목 · 데이터 기준일 {loading.asof(px)})")
    return fails


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    total = 0
    if which in ("kr", "all"):
        total += run_kr()
    if which in ("us", "all"):
        total += run_us()
    sys.exit(1 if total else 0)
