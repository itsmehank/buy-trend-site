"""가격 이력 수집 — 미국(yfinance) / 한국(pykrx, 수정주가) + parquet 증분 캐시."""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
import yfinance as yf

from . import config
from . import krx  # noqa: F401 — import 부작용(타임아웃 가드) 필요, pykrx보다 먼저

log = logging.getLogger(__name__)

COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def _cache_path(market: str):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"prices_{market}.parquet"


def load_cache(market: str) -> pd.DataFrame:
    p = _cache_path(market)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=COLS)


def save_cache(market: str, df: pd.DataFrame) -> None:
    df = (df.drop_duplicates(subset=["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    df.to_parquet(_cache_path(market), index=False)


def _history_start() -> dt.date:
    return dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25))


def _trim_asof(df: pd.DataFrame, asof: dt.date | None) -> pd.DataFrame:
    """asof(완결 거래일) 이후의 미완결 봉 제거. asof=None이면 트림 안 함."""
    if asof is None or df.empty:
        return df
    return df[df["date"] <= asof]


def fetch_us(tickers: list[str], market: str = "us", chunk: int = 100,
             asof: dt.date | None = None) -> pd.DataFrame:
    """yfinance 일괄 다운로드(수정주가) + 증분 캐시. market: 캐시 키(us/etf/bench).

    asof: 이 날짜 이후의 미완결 봉(장중 오늘 봉 등)을 캐시·반환에서 제외한다.
    """
    cache = load_cache(market)
    last_dates = cache.groupby("ticker")["date"].max() if len(cache) else pd.Series(dtype="object")
    today = dt.date.today()
    frames = [cache]
    todo = list(tickers)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        # 배치 내 가장 오래된 시작점 기준으로 다운로드 (신규 티커는 전체 이력)
        starts = [pd.Timestamp(last_dates.get(t, pd.Timestamp(_history_start()))) for t in batch]
        start = min(starts).date()
        raw = yf.download(batch, start=start, end=today + dt.timedelta(days=1),
                          auto_adjust=True, group_by="ticker", threads=True,
                          progress=False)
        if raw is None or raw.empty:
            continue
        rows = []
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            rows.append(pd.DataFrame({
                "ticker": t,
                "date": pd.to_datetime(sub.index).date,
                "open": sub["Open"].values, "high": sub["High"].values,
                "low": sub["Low"].values, "close": sub["Close"].values,
                "volume": sub["Volume"].values,
            }))
        if rows:
            frames.append(pd.concat(rows, ignore_index=True))
        log.info("US fetch %d/%d", min(i + chunk, len(todo)), len(todo))
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    df = _trim_asof(df, asof)  # 미완결 봉 제거 후 캐시 (캐시 오염 방지)
    save_cache(market, df)
    return df[df["ticker"].isin(tickers)]


def fetch_kr(tickers: list[str], market: str = "kr", pause: float = 0.2,
             asof: dt.date | None = None) -> pd.DataFrame:
    """pykrx 수정주가 OHLCV + 증분 캐시. asof 이후 미완결 봉 제외."""
    from pykrx import stock as krx

    cache = load_cache(market)
    last_dates = cache.groupby("ticker")["date"].max() if len(cache) else pd.Series(dtype="object")
    today = dt.date.today()
    frames = [cache]
    for i, t in enumerate(tickers):
        start = last_dates.get(t)
        start = (pd.Timestamp(start) + pd.Timedelta(days=1)).date() if start is not None else _history_start()
        if start > today:
            continue
        try:
            ohlcv = krx.get_market_ohlcv(start.strftime("%Y%m%d"),
                                         today.strftime("%Y%m%d"), t, adjusted=True)
        except Exception as e:  # 개별 티커 실패는 건너뜀
            log.warning("KR fetch fail %s: %s", t, e)
            continue
        if ohlcv is None or ohlcv.empty:
            continue
        frames.append(pd.DataFrame({
            "ticker": t,
            "date": pd.to_datetime(ohlcv.index).date,
            "open": ohlcv["시가"].values, "high": ohlcv["고가"].values,
            "low": ohlcv["저가"].values, "close": ohlcv["종가"].values,
            "volume": ohlcv["거래량"].values,
        }))
        if pause:
            time.sleep(pause)
        if (i + 1) % 100 == 0:
            log.info("KR fetch %d/%d", i + 1, len(tickers))
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    df = _trim_asof(df, asof)  # 장중 미완결 봉 제거 후 캐시
    save_cache(market, df)
    return df[df["ticker"].isin(tickers)]


def fetch_benchmarks() -> dict[str, pd.DataFrame]:
    """^GSPC, ^KS11 이력. 반환: {symbol: df(date, close)}."""
    df = fetch_us(list(config.BENCHMARKS.values()), market="bench")
    return {sym: df[df["ticker"] == sym].sort_values("date").reset_index(drop=True)
            for sym in config.BENCHMARKS.values()}


def usdkrw() -> float:
    """원/달러 환율 (시총 USD 환산용). 실패 시 보수적 기본값."""
    try:
        h = yf.Ticker("KRW=X").history(period="5d")
        return float(h["Close"].dropna().iloc[-1])
    except Exception:
        return 1400.0


def to_ticker_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long 포맷 → {ticker: 날짜순 df}. 최소 60행 미만 티커는 제외."""
    out = {}
    for t, sub in df.groupby("ticker"):
        sub = sub.sort_values("date").reset_index(drop=True)
        if len(sub) >= 60:
            out[t] = sub
    return out
