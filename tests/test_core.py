"""수치 로직 단위 테스트 — EV, 최적 기간 선정(§2.4 검증사례), zone, 별점, 터치·돌파 정의."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pipeline import backtest, phrase, signals, stars, trading, zone
from pipeline.rs import rs_percentiles, rs_raw


# ── EV 공식 (§2.4)

def test_ev_formula_matches_rsi_row():
    # 명세서 실측: RSI 126일 — win 91.25 / avg_win 27.4 / avg_loss -2.41 → ev ≈ 10.27
    e = backtest.ev(91.25, 27.4, -2.41)
    assert e == pytest.approx(10.2694, abs=0.05)


def test_ev_undefined_without_loss():
    assert backtest.ev(100.0, 10.0, None) is None
    assert backtest.ev(100.0, 10.0, 0) is None


# ── 최적 보유기간 선정 (§2.4 검증사례)

def test_optimal_excludes_no_loss_period():
    # RSI 사례: 252일이 승률 100%(표본 20, 무손실)여도 EV 미정의 → 126일 선택
    rows = [
        {"hold": 126, "win_rate": 91.25, "avg_win": 27.4, "avg_loss": -2.41,
         "pl_ratio": 11.37, "n": 23},
        {"hold": 252, "win_rate": 100.0, "avg_win": 55.0, "avg_loss": None,
         "pl_ratio": None, "n": 20},
    ]
    best = backtest.select_optimal(rows)
    assert best["hold"] == 126


def test_optimal_excludes_small_sample():
    # ESTA 사례: 5일 EV 0.77 > 3일 EV 0.71이지만 표본 15 < 20 → 3일 선택
    rows = [
        {"hold": 3, "win_rate": 55.0, "avg_win": 4.0, "avg_loss": -1.62,
         "pl_ratio": 2.47, "n": 20},   # EV ≈ 0.71
        {"hold": 5, "win_rate": 55.0, "avg_win": 4.3, "avg_loss": -1.6,
         "pl_ratio": 2.69, "n": 15},   # EV ≈ 0.77 — 표본 미달
    ]
    best = backtest.select_optimal(rows)
    assert best["hold"] == 3
    assert best["ev"] == pytest.approx(0.908, abs=0.005)


def test_optimal_none_when_no_candidate():
    rows = [{"hold": 2, "win_rate": 60.0, "avg_win": 2.0, "avg_loss": -1.0,
             "pl_ratio": 2.0, "n": 5}]
    assert backtest.select_optimal(rows) is None


def test_hold_stats_basic():
    closes = np.array([100, 110, 121, 100, 90, 99], dtype=float)
    s = backtest.hold_stats(closes, np.array([0, 3]), 1)
    # 0→1: +10% 승, 3→4: -10% 패
    assert s["n"] == 2
    assert s["win_rate"] == 50.0
    assert s["avg_win"] == pytest.approx(10.0)
    assert s["avg_loss"] == pytest.approx(-10.0)
    assert s["pl_ratio"] == pytest.approx(1.0)


def test_hold_stats_drops_events_without_future():
    closes = np.array([100, 110, 121], dtype=float)
    s = backtest.hold_stats(closes, np.array([0, 2]), 2)  # idx 2는 미래 없음
    assert s["n"] == 1


# ── 매수 구간 (§2.5)

def test_zone_ma():
    low, high = zone.buy_zone("이평", 32.62)
    assert low == pytest.approx(32.62 * 0.98)
    assert high == pytest.approx(32.62 * 1.03)


def test_zone_breakout():
    for sig in ("박스", "신고가"):
        low, high = zone.buy_zone(sig, 100.0)
        assert low == 100.0
        assert high == pytest.approx(105.0)


def test_in_zone():
    assert zone.in_zone("이평", 32.62, 32.5)          # 명세서 RSI 실측
    assert not zone.in_zone("이평", 32.62, 30.0)
    assert zone.in_zone("신고가", 100.0, 104.9)
    assert not zone.in_zone("신고가", 100.0, 99.9)


# ── 별점 (§2.6)

def test_stars_default_three():
    s, cut = stars.stars_from_tags([], "stock")
    assert (s, cut) == (3, None)


def test_stars_minor_tags_two():
    s, cut = stars.stars_from_tags([stars.TAG_WICK, stars.TAG_RS_HOT], "stock")
    assert s == 2
    assert cut == "윗꼬리 5일+ / RS최고 97+(초과열)"


def test_stars_surge_one():
    s, _ = stars.stars_from_tags([stars.TAG_VOL_Q4, stars.TAG_SURGE], "stock")
    assert s == 1


def test_stars_etf_null():
    s, _ = stars.stars_from_tags([stars.TAG_WICK], "etf")
    assert s is None


def test_surge_flags():
    base = np.full(100, 100.0)
    surged = base.copy()
    surged[-1] = 155.0  # 하루 +55% → 20일 +50% & 일간 +20% 모두 충족
    assert stars.surge_flags(surged)
    assert not stars.surge_flags(base)


def test_upper_wick_days():
    n = 10
    opens = np.full(n, 100.0)
    closes = np.full(n, 101.0)          # 몸통 1
    highs = np.full(n, 101.5)           # 윗꼬리 0.5 < 몸통
    highs[:5] = 104.0                   # 5일은 윗꼬리 3 > 몸통
    assert stars.upper_wick_days(opens, highs, closes) == 5


# ── 터치 정의 ([자체 기준] 4)

def test_touch_counts_entry_only():
    near = pd.Series([False, True, True, True, False, True, True])
    ctx = pd.Series([True] * 7)
    ev = signals.touch_events(near, ctx)
    assert list(ev) == [1, 5]  # 연속 근접은 1회, 재진입 시 새 터치


def test_touch_requires_context():
    near = pd.Series([False, True, False, True])
    ctx = pd.Series([True, False, True, True])
    assert list(signals.touch_events(near, ctx)) == [3]


# ── 신고가 크로싱 dedup

def test_nhigh_crossing_once_per_run():
    # 55→60→61→62: 60에서 크로싱 1회, 이후 연속 신고가는 미카운트
    closes = pd.Series([50, 51, 52, 55, 54, 53, 60, 61, 62], dtype=float)
    idx, lines = signals.nhigh_breakouts(closes, 5)
    assert 6 in idx
    diffs = np.diff(idx)
    assert (diffs > 1).all() or len(idx) <= 1


def test_nhigh_ath_line_is_prior_max():
    closes = pd.Series([10, 20, 15, 25], dtype=float)
    idx, lines = signals.nhigh_breakouts(closes, "ATH")
    assert list(idx) == [1, 3]
    assert lines[1] == pytest.approx(20.0)  # 돌파선 = 직전 역대최고


# ── 박스 돌파 ([자체 기준] 3)

def _box_series(box_days=25, top=105.0, bot=100.0):
    highs = [top] * box_days
    lows = [bot] * box_days
    closes = [102.0] * box_days
    return highs, lows, closes


def test_box_breakout_detected():
    highs, lows, closes = _box_series()
    highs += [110.0]; lows += [106.0]; closes += [109.0]  # 상단(105) 돌파
    idx, tops = signals.box_breakouts(
        pd.Series(highs, dtype=float), pd.Series(lows, dtype=float),
        pd.Series(closes, dtype=float), 20)
    assert len(idx) == 1
    assert tops[0] == pytest.approx(105.0)


def test_box_no_rebreakout_without_reformation():
    highs, lows, closes = _box_series()
    # 돌파 후 계속 상승 — 박스 재형성 없이 연속 돌파 조건이 성립해도 1회만
    for px in (109.0, 111.0, 113.0):
        highs.append(px + 1); lows.append(px - 1); closes.append(px)
    idx, _ = signals.box_breakouts(
        pd.Series(highs, dtype=float), pd.Series(lows, dtype=float),
        pd.Series(closes, dtype=float), 20)
    assert len(idx) == 1


def test_box_amplitude_filter():
    # 진폭 20% > 15% → 박스 아님
    highs = [120.0] * 25 + [125.0]
    lows = [100.0] * 25 + [121.0]
    closes = [110.0] * 25 + [124.0]
    idx, _ = signals.box_breakouts(
        pd.Series(highs, dtype=float), pd.Series(lows, dtype=float),
        pd.Series(closes, dtype=float), 20)
    assert len(idx) == 0


# ── RS ([자체 기준] 1)

def test_rs_raw_weights():
    n = 260
    closes = np.linspace(100, 200, n)  # 단조 상승
    raw = rs_raw(closes)
    expected = sum(w * (closes[-1] / closes[-1 - lb] - 1.0)
                   for lb, w in {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2}.items())
    assert raw == pytest.approx(expected)


def test_rs_raw_needs_full_history():
    assert rs_raw(np.linspace(100, 200, 100)) is None


def test_rs_percentiles_range():
    raws = {f"T{i}": float(i) for i in range(200)}
    pct = rs_percentiles(raws)
    assert pct["T199"] == 100
    assert pct["T0"] <= 1
    assert all(0 <= v <= 100 for v in pct.values())


# ── phrase (§2.2)

def test_phrase_ma():
    assert phrase.make_phrase("이평", "L/EMA10", 126) == "EMA10 장기 근접 · 126일 보유"
    assert phrase.make_phrase("이평", "S/SMA20", 5) == "SMA20 단기 근접 · 5일 보유"


def test_phrase_box_and_nhigh():
    assert phrase.make_phrase("박스", "L", 63) == "박스돌파 장기 · 63일 보유"
    assert phrase.make_phrase("신고가", "ATH", 20) == "역대최고 돌파 · 20일 보유"
    assert phrase.make_phrase("신고가", "55", 10) == "55일신고가 돌파 · 10일 보유"


# ── 완결 거래일 가드 (장중 미완결 봉 배제)

def _utc(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)


def test_kr_intraday_excludes_today():
    # 2026-07-22 11:48 KST (=02:48 UTC), KR 장중 → 완결일은 전일 07-21
    assert trading.latest_complete_date("KR", _utc(2026, 7, 22, 2, 48)) == dt.date(2026, 7, 21)


def test_kr_after_close_includes_today():
    # 2026-07-22 16:30 KST (=07:30 UTC), 장 마감+버퍼 후 → 당일 07-22
    assert trading.latest_complete_date("KR", _utc(2026, 7, 22, 7, 30)) == dt.date(2026, 7, 22)


def test_us_intraday_excludes_today():
    # 2026-07-22 14:00 ET (=18:00 UTC, EDT), US 장중 → 전일 07-21
    assert trading.latest_complete_date("US", _utc(2026, 7, 22, 18, 0)) == dt.date(2026, 7, 21)


def test_us_after_close_includes_today():
    # 2026-07-22 17:00 ET (=21:00 UTC, EDT), 마감+버퍼 후 → 당일 07-22
    assert trading.latest_complete_date("US", _utc(2026, 7, 22, 21, 0)) == dt.date(2026, 7, 22)


# ── rs_asof = 실제 마지막 봉 날짜

def test_actual_asof_uses_last_bar_not_expected_date():
    from pipeline.build import actual_asof
    frames = {
        "A": pd.DataFrame({"date": [dt.date(2026, 7, 20), dt.date(2026, 7, 21)]}),
        "B": pd.DataFrame({"date": [dt.date(2026, 7, 17)]}),
    }
    # 기대 완결일이 토요일(07-25)이어도 실제 마지막 봉(07-21)을 써야 한다
    assert actual_asof(frames, "2026-07-25") == "2026-07-21"


def test_actual_asof_falls_back_when_no_frames():
    from pipeline.build import actual_asof
    assert actual_asof({}, "2026-07-25") == "2026-07-25"


# ── 백테스트 구간 표기 (표본수 해석용)

def _span_frame(start: dt.date, n: int) -> pd.DataFrame:
    return pd.DataFrame({"date": [start + dt.timedelta(days=i) for i in range(n)]})


def test_history_span_reports_actual_range_and_bar_spread():
    from pipeline.build import history_span
    frames = {
        "OLD": _span_frame(dt.date(2011, 7, 1), 100),   # 오래된 종목
        "MID": _span_frame(dt.date(2015, 1, 1), 50),
        "NEW": _span_frame(dt.date(2024, 1, 1), 10),    # 최근 상장
    }
    s = history_span(frames)
    assert s["start"] == "2011-07-01"                   # 가장 이른 첫 봉
    assert s["end"] == "2024-01-10"                     # 가장 늦은 마지막 봉
    assert (s["bars_min"], s["bars_median"], s["bars_max"]) == (10, 50, 100)
    assert s["tickers"] == 3


def test_history_span_ignores_empty_frames():
    from pipeline.build import history_span
    frames = {
        "A": _span_frame(dt.date(2020, 1, 1), 5),
        "EMPTY": pd.DataFrame({"date": []}),
    }
    s = history_span(frames)
    assert s["tickers"] == 1 and s["bars_min"] == 5


def test_history_span_empty_when_no_data():
    from pipeline.build import history_span
    assert history_span({}) == {}


# ── smoke: bars는 '이번에 만든 row'에만 요구한다
# (갱신 안 한 시장의 row는 기존 JSON에서 그대로 이어받아 bars가 없다)

def test_bars_not_required_on_carried_over_rows():
    from pipeline.smoke import REQUIRED_ROW_FIELDS
    assert "bars" not in REQUIRED_ROW_FIELDS


def test_fresh_rows_must_carry_bars_and_n_eff():
    from pipeline.smoke import _validate_fresh_rows
    ok = {"ticker": "A", "bars": 3776, "n": 60, "n_eff": 8}
    assert _validate_fresh_rows([ok]) == []

    assert _validate_fresh_rows([{"ticker": "B", "n": 60, "n_eff": 8}]) \
        == ["B: 신규 row에 bars 누락"]
    assert _validate_fresh_rows([{"ticker": "C", "bars": 3776, "n": 60}]) \
        == ["C: 신규 row에 n_eff 누락"]

    # n_eff는 겹침을 제거한 값이라 n을 넘을 수 없다
    over = _validate_fresh_rows([{**ok, "ticker": "D", "n": 60, "n_eff": 61}])
    assert len(over) == 1 and "범위 밖" in over[0]
    assert len(_validate_fresh_rows([{**ok, "ticker": "E", "bars": 0},
                                     {**ok, "ticker": "F", "n_eff": 0}])) == 2


# ── 독립 표본수 (겹치는 보유구간 제거) — 표시 전용

def test_independent_count_removes_overlap():
    from pipeline.backtest import independent_count
    # 0,1,2일에 신호 → 10일 보유면 구간이 거의 겹쳐 독립은 1회뿐
    assert independent_count(np.array([0, 1, 2]), 10, 100) == 1
    # 0,10,20 → 정확히 맞닿으면 겹치지 않음
    assert independent_count(np.array([0, 10, 20]), 10, 100) == 3
    # 0,9,10 → 9는 0의 구간(0~10)과 겹쳐 탈락, 10은 채택
    assert independent_count(np.array([0, 9, 10]), 10, 100) == 2


def test_independent_count_bounded_by_history_length():
    from pipeline.backtest import independent_count
    # 3780거래일(15년)에서 매일 신호가 떠도 252일 보유 독립 표본은 14회가 상한.
    # 나눗셈 근사(3780/252=15)가 아니라 14인 이유: 마지막 구간은 252일 뒤
    # 종가가 없어 완결되지 않으므로 표본에서 빠진다(hold_stats와 같은 기준).
    every_day = np.arange(3780)
    assert independent_count(every_day, 252, 3780) == 14
    assert independent_count(every_day, 2, 3780) == 1889


def test_independent_count_excludes_events_without_future():
    from pipeline.backtest import independent_count
    # 95일차 이벤트는 10일 뒤 데이터가 없어 hold_stats와 동일하게 제외
    assert independent_count(np.array([0, 95]), 10, 100) == 1
    assert independent_count(np.array([]), 10, 100) == 0


def test_hold_stats_reports_n_eff_below_n():
    closes = np.linspace(100, 200, 500)
    events = np.arange(0, 200)          # 200개 이벤트, 전부 겹침
    s = backtest.hold_stats(closes, events, 100)
    assert s["n"] == 200                # 표시되는 표본수
    assert s["n_eff"] == 2              # 실제 독립 관측
    assert s["n_eff"] < s["n"]


# n_eff를 선정에 쓰면 명세서 ✅(min_sample=20 + EV 최대)를 깨뜨린다 — 표시 전용 고정
def test_select_optimal_ignores_n_eff():
    rows = [
        {"hold": 2, "win_rate": 50.0, "avg_win": 4.0, "avg_loss": -2.0,
         "pl_ratio": 2.0, "n": 25, "n_eff": 25},
        # 독립 표본은 1개뿐이지만 n=25라 명세서상 후보 자격이 있고 EV가 더 높다
        {"hold": 252, "win_rate": 90.0, "avg_win": 40.0, "avg_loss": -2.0,
         "pl_ratio": 20.0, "n": 25, "n_eff": 1},
    ]
    best = backtest.select_optimal(rows)
    assert best["hold"] == 252, "선정은 n_eff와 무관하게 n 기준이어야 한다"
