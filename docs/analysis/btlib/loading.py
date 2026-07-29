"""로컬 parquet 캐시 로딩 — 재다운로드 없이 사용 (PROTOCOL §1).

캐시 갱신은 사용자 몫이다. 마지막 봉이 6개월 이상 오래되면 경고 문자열을 돌려준다.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from pipeline import config

CACHE_FILES = {"kr": "prices_kr.parquet", "us": "prices_us_test.parquet",
               "bench": "prices_bench.parquet"}
BENCH_SYMBOL = {"us": "^GSPC", "kr": "^KS11"}


def load_prices(market: str) -> pd.DataFrame:
    return pd.read_parquet(config.CACHE_DIR / CACHE_FILES[market])


def load_bench(market: str) -> pd.DataFrame:
    df = pd.read_parquet(config.CACHE_DIR / CACHE_FILES["bench"])
    return (df[df["ticker"] == BENCH_SYMBOL[market]][["date", "close"]]
            .sort_values("date").reset_index(drop=True))


def asof(df: pd.DataFrame):
    """데이터 기준일 = 마지막 봉 날짜. 결과 문서에 반드시 기록할 것."""
    return df["date"].max()


def staleness_warning(df: pd.DataFrame, months: int = 6) -> str | None:
    last = asof(df)
    if dt.date.today() - last > dt.timedelta(days=months * 30):
        return (f"캐시 갱신 권고: 마지막 봉이 {last}로 {months}개월 이상 지났다. "
                "갱신 방법은 docs/analysis/PROTOCOL.md §1 참조.")
    return None
