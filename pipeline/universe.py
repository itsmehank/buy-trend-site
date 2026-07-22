"""[자체 기준] 7: 유니버스 구성.

- 미국 주식: NASDAQ 스크리너(무키 공개 API) → 시총 ≥ $300M, 이후 20일 평균
  거래대금 ≥ $2M은 가격 다운로드 후 확정 필터.
- 한국: pykrx 시가총액 → 3000억원 이상 (KOSPI+KOSDAQ).
- ETF: NASDAQ 스크리너 ETF 목록 → 20일 평균 거래대금 상위 500
  (AUM 공개 API가 없어 거래대금을 AUM 근사 프록시로 사용, README 명시).
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

NASDAQ_STOCKS = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
NASDAQ_ETF = "https://api.nasdaq.com/api/screener/etf?download=true"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
           "Accept": "application/json"}


def _get_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def us_stock_universe() -> pd.DataFrame:
    """columns: ticker, name, market_cap_usd, sector, industry."""
    rows = _get_json(NASDAQ_STOCKS)["data"]["rows"]
    df = pd.DataFrame(rows)
    df["market_cap_usd"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df["lastsale"] = pd.to_numeric(df["lastsale"].str.replace("[$,]", "", regex=True),
                                   errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df[df["market_cap_usd"] >= config.US_MIN_MCAP_USD]
    # 1차 거래대금 프록시(당일) — 최종 20일 평균 필터는 다운로드 후 적용
    df = df[df["lastsale"] * df["volume"] >= config.US_MIN_DOLLAR_VOL * 0.5]
    df = df[~df["symbol"].str.contains(r"[\^/]", na=True)]
    df["ticker"] = df["symbol"].str.strip().str.replace(".", "-", regex=False)
    out = df[["ticker", "name", "market_cap_usd", "sector", "industry"]].copy()
    out["sector"] = out["sector"].replace("", None)
    out["industry"] = out["industry"].replace("", None)
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def us_etf_universe() -> pd.DataFrame:
    """columns: ticker, name. (상위 500 선별은 가격 다운로드 후 거래대금 기준)"""
    rows = _get_json(NASDAQ_ETF)["data"]["data"]["rows"]
    df = pd.DataFrame(rows)
    df["ticker"] = df["symbol"].str.strip().str.replace(".", "-", regex=False)
    df = df[~df["ticker"].str.contains(r"[\^/]", na=True)]
    return (df[["ticker", "companyName"]].rename(columns={"companyName": "name"})
            .drop_duplicates("ticker").reset_index(drop=True))


def kr_universe(usdkrw_rate: float) -> pd.DataFrame:
    """columns: ticker, name, market_cap_usd (KOSPI+KOSDAQ, 시총 ≥ 3000억)."""
    from pykrx import stock as krx

    day = dt.date.today()
    cap = pd.DataFrame()
    for _ in range(10):  # 최근 영업일 탐색
        cap = krx.get_market_cap(day.strftime("%Y%m%d"), market="ALL")
        if cap is not None and len(cap) and cap["시가총액"].sum() > 0:
            break
        day -= dt.timedelta(days=1)
    cap = cap[cap["시가총액"] >= config.KR_MIN_MCAP_KRW]
    tickers = list(cap.index)
    names = {t: krx.get_market_ticker_name(t) for t in tickers}
    return pd.DataFrame({
        "ticker": tickers,
        "name": [names[t] for t in tickers],
        "market_cap_usd": (cap["시가총액"] / usdkrw_rate).round().astype("int64").values,
    }).reset_index(drop=True)


def dollar_vol_filter(frames: dict[str, pd.DataFrame], min_dollar_vol: float) -> list[str]:
    """20일 평균 거래대금(종가×거래량) ≥ min_dollar_vol 티커만."""
    keep = []
    for t, df in frames.items():
        tail = df.tail(20)
        if (tail["close"] * tail["volume"]).mean() >= min_dollar_vol:
            keep.append(t)
    return keep


def top_etfs_by_dollar_vol(frames: dict[str, pd.DataFrame], n: int = config.ETF_TOP_N) -> list[str]:
    dv = {t: float((df.tail(20)["close"] * df.tail(20)["volume"]).mean())
          for t, df in frames.items()}
    return [t for t, _ in sorted(dv.items(), key=lambda kv: -kv[1])[:n]]


def save_universe(name: str, df: pd.DataFrame) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(config.CACHE_DIR / f"universe_{name}.json", orient="records", force_ascii=False)


def load_universe(name: str) -> pd.DataFrame | None:
    p = config.CACHE_DIR / f"universe_{name}.json"
    if p.exists():
        return pd.read_json(p, dtype={"ticker": str})
    return None
