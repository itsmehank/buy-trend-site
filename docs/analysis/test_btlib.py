"""btlib(공용 백테스트 하네스) 단위 테스트 — 합성 데이터만 사용, 캐시·네트워크 불필요.

핵심 검증 대상은 2026-07-29 검증에서 실제로 버그가 났던 지점이다:
- E 규칙(연속 하회)이 이벤트 이전에 시작된 연속을 이어받지 않는가 (§5-①)
- 유동성 필터가 진입 시점 기준(PIT)인가 (§5-②)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # docs/analysis
ROOT = HERE.parents[1]                          # 저장소 루트 (pipeline import용)
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from btlib import costs, engine, entries, exits, liquidity, metrics, regime  # noqa: E402


# ── exits ──────────────────────────────────────────────────────────────

def test_trailing_stop_fires_on_known_day():
    seg = np.array([100.0, 110.0, 120.0, 101.0, 90.0])
    assert exits.trailing_days(seg, 0.85, maxh=4) == 3   # 101 <= 120*0.85


def test_trailing_stop_never_fires_returns_maxh():
    seg = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    assert exits.trailing_days(seg, 0.85, maxh=4) == 4


def test_sma_below_returns_first_day_after_entry():
    below_idx = np.array([2, 7])
    assert exits.below_sma_days(below_idx, i=1, maxh=10) == 1   # 다음날 바로 하회
    assert exits.below_sma_days(below_idx, i=2, maxh=10) == 5   # 다음 하회는 7
    assert exits.below_sma_days(below_idx, i=7, maxh=10) == 10  # 이후 없음 → maxh


def test_run_lengths_counts_consecutive_true():
    below = np.array([False, True, True, False, True, True, True])
    assert list(exits.run_lengths(below)) == [0, 1, 2, 0, 1, 2, 3]


def test_consec_below_ignores_run_started_before_event():
    # 이벤트(i=2) 이전부터 시작된 연속 하회를 이어받으면 1~4일에 조기 발동한다.
    # 이벤트 이후부터 새로 세면 i+1..i+5(5일 연속)에서 정확히 5일째 발동해야 한다.
    below = np.array([True] * 12)
    runlen = exits.run_lengths(below)
    assert exits.consec_below_days(runlen, i=2, n_days=5, maxh=9) == 5


def test_consec_below_matches_loop_reference_on_random_data():
    rng = np.random.default_rng(0)
    below = rng.random(400) < 0.5
    runlen = exits.run_lengths(below)
    maxh = 60
    for i in range(0, 300, 7):
        # 원본 KR 스크립트와 같은 루프 기준 구현
        run, expected = 0, maxh
        for k in range(1, maxh + 1):
            run = run + 1 if below[i + k] else 0
            if run >= 5:
                expected = k
                break
        assert exits.consec_below_days(runlen, i=i, n_days=5, maxh=maxh) == expected, i


# ── liquidity (PIT) ────────────────────────────────────────────────────

def test_pit_dollar_vol_uses_only_trailing_window():
    close = pd.Series(np.full(30, 10.0))
    volume = pd.Series(np.concatenate([np.full(25, 100.0), np.full(5, 10_000.0)]))
    dv = liquidity.pit_dollar_vol(close, volume, window=20)
    assert np.isnan(dv[18])                       # 워밍업 구간
    assert dv[19] == 10.0 * 100.0                 # 과거 20봉 평균만 사용
    assert dv[24] == 1000.0                       # 뒤의 급증(idx 25~)이 새어들지 않음


# ── regime ─────────────────────────────────────────────────────────────

def test_bull_map_true_only_above_sma():
    dates = pd.date_range("2020-01-01", periods=6).date
    close = [10.0, 10.0, 10.0, 20.0, 20.0, 1.0]
    bench = pd.DataFrame({"date": dates, "close": close})
    m = regime.bull_map(bench, sma=3)
    assert m[dates[1]] is False                   # SMA 워밍업 → False(보수적)
    assert m[dates[3]] is True                    # 20 > mean(10,10,20)
    assert m[dates[5]] is False                   # 1 < mean


# ── metrics / costs ────────────────────────────────────────────────────

def test_summarize_known_values():
    s = metrics.summarize(exc=[0.10, -0.05], days=[10, 20])
    assert s["n"] == 2
    assert s["mean_pct"] == 2.5
    assert s["median_pct"] == 2.5
    assert s["winrate_pct"] == 50.0
    assert s["avg_days"] == 15.0
    tpy = 252 / 15.0
    assert abs(s["net_ann_pct"]["0.10"] - (2.5 * tpy - tpy * 0.10)) < 1e-9
    assert abs(s["net_ann_pct"]["0.28"] - (2.5 * tpy - tpy * 0.28)) < 1e-9


def test_market_costs():
    assert costs.COST_PCT["US"] == 0.10
    assert costs.COST_PCT["KR"] == 0.28


# ── entries ────────────────────────────────────────────────────────────

def _frame(close, high=None, low=None, volume=None):
    n = len(close)
    dates = pd.date_range("2015-01-01", periods=n).date
    return pd.DataFrame({
        "ticker": "T", "date": dates,
        "open": close, "high": high if high is not None else close,
        "low": low if low is not None else close, "close": close,
        "volume": volume if volume is not None else np.full(n, 1000.0),
    })


def test_entry_nhigh_matches_pipeline_signals():
    from pipeline import signals
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 300)))
    g = _frame(close.to_numpy())
    got = entries.resolve_entry(("신고가", 20))(g)
    want = signals.nhigh_breakouts(close.reset_index(drop=True), 20)[0]
    assert np.array_equal(got, want)


def test_entry_sma_cross_above_fires_on_crossing_day():
    close = np.array([10.0] * 10 + [5.0, 5.0, 5.0, 12.0, 13.0])
    g = _frame(close)
    ev = entries.resolve_entry(("이평선 상향돌파", 3))(g)
    assert 13 in ev                               # 12 > mean(5,5,12)? 아니면 13에서 판정
    # 돌파일 이후 연속 상회는 이벤트가 아니다 (크로싱만)
    assert len(ev) >= 1
    assert not (set(ev) >= {13, 14})


# ── engine ─────────────────────────────────────────────────────────────

def _tiny_universe(n=80):
    close = np.linspace(100.0, 179.0, n)          # 단조 상승
    px = _frame(close)
    dates = px["date"]
    bench = pd.DataFrame({"date": dates, "close": np.full(n, 50.0)})  # 지수 무변동
    return px, bench


def test_engine_single_trade_excess_and_days():
    px, bench = _tiny_universe()
    res = engine.run_event_backtest(
        px, bench,
        entry_specs={"E": lambda g: np.array([10])},
        exit_specs={"X": ("fixed", 5)},
        min_bars=50, maxh=10)
    tr = res["E"]["X"]
    c = px["close"].to_numpy()
    assert tr["days"] == [5]
    assert abs(tr["exc"][0] - (c[15] / c[10] - 1.0)) < 1e-12   # 지수 0% 차감


def test_engine_regime_filter_skips_bear_entries():
    px, bench = _tiny_universe()
    bear = {d: False for d in px["date"]}
    res = engine.run_event_backtest(
        px, bench, entry_specs={"E": lambda g: np.array([10])},
        exit_specs={"X": ("fixed", 5)}, min_bars=50, maxh=10, regime_map=bear)
    assert res["E"]["X"]["exc"] == []


def test_engine_liquidity_filter_skips_thin_entries():
    px, bench = _tiny_universe()
    res = engine.run_event_backtest(
        px, bench, entry_specs={"E": lambda g: np.array([30])},
        exit_specs={"X": ("fixed", 5)}, min_bars=50, maxh=10,
        min_dollar_vol=1e12)                      # 통과 불가능한 하한
    assert res["E"]["X"]["exc"] == []


def test_engine_skips_nonpositive_close_ticker():
    px, bench = _tiny_universe()
    px.loc[3, "close"] = 0.0                      # CBIO/DEC 유형
    res = engine.run_event_backtest(
        px, bench, entry_specs={"E": lambda g: np.array([10])},
        exit_specs={"X": ("fixed", 5)}, min_bars=50, maxh=10)
    assert res["E"]["X"]["exc"] == []


def test_engine_exit_rule_specs_resolve():
    px, bench = _tiny_universe()
    res = engine.run_event_backtest(
        px, bench, entry_specs={"E": lambda g: np.array([10])},
        exit_specs={"A": ("fixed", 252), "C": ("trailing", 0.85),
                    "D": ("sma_below", 60), "E5": ("consec_below", (20, 5))},
        min_bars=50, maxh=10)
    # 단조 상승 + maxh=10 → 전 규칙 maxh(또는 252→bp 범위 밖 제외) 처리돼도 크래시 없음
    assert set(res["E"].keys()) == {"A", "C", "D", "E5"}
