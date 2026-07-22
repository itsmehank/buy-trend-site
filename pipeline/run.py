"""CLI 진입점 — 시장별 일간 배치.

사용:
  python -m pipeline.run --market us      # 미국 주식+ETF 갱신
  python -m pipeline.run --market kr      # 한국 주식 갱신
  python -m pipeline.run --market all     # 둘 다
  python -m pipeline.run --smoke          # 소량 티커 end-to-end (네트워크만)

각 실행은 해당 시장 데이터만 내려받아 build_market → merge_and_write로
기존 JSON과 병합한다(부분 갱신). 실패한 시장은 직전 JSON을 유지한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import pandas as pd

from . import build, config, data, universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bench_frames():
    b = data.fetch_benchmarks()
    return b


def run_us(now_iso: str, built: dict, regimes: dict, fresh: dict):
    log.info("── US universe")
    stocks = universe.us_stock_universe()
    etfs = universe.us_etf_universe()
    universe.save_universe("us_stock", stocks)
    universe.save_universe("us_etf", etfs)

    log.info("US stocks: %d, ETFs: %d — downloading prices", len(stocks), len(etfs))
    sdf = data.fetch_us(stocks["ticker"].tolist(), market="us")
    edf = data.fetch_us(etfs["ticker"].tolist(), market="etf")
    s_frames = data.to_ticker_frames(sdf)
    e_frames = data.to_ticker_frames(edf)

    # 거래대금 필터
    keep_s = set(universe.dollar_vol_filter(s_frames, config.US_MIN_DOLLAR_VOL))
    s_frames = {t: f for t, f in s_frames.items() if t in keep_s}
    keep_e = set(universe.top_etfs_by_dollar_vol(e_frames))
    e_frames = {t: f for t, f in e_frames.items() if t in keep_e}
    log.info("after dollar-vol filter: stocks %d, ETFs %d", len(s_frames), len(e_frames))

    frames = {**s_frames, **e_frames}
    asset_map = {**{t: "stock" for t in s_frames}, **{t: "etf" for t in e_frames}}
    name_map = dict(zip(stocks["ticker"], stocks["name"]))
    name_map.update(dict(zip(etfs["ticker"], etfs["name"])))
    sector_map = dict(zip(stocks["ticker"], stocks["sector"]))
    industry_map = dict(zip(stocks["ticker"], stocks["industry"]))
    mcap_map = dict(zip(stocks["ticker"], stocks["market_cap_usd"]))

    return _finish_market("US", frames, {
        "country": "US", "asset_map": asset_map, "name_map": name_map,
        "sector_map": sector_map, "industry_map": industry_map,
        "category_map": {}, "mcap_map": mcap_map,
        "benchmark": "^GSPC", "market_s": regimes["US"],
        "rs_asof": str(dt.date.today()),
    }, built)


def run_kr(now_iso: str, built: dict, regimes: dict, fresh: dict, rate: float):
    log.info("── KR universe")
    uni = universe.kr_universe(rate)
    universe.save_universe("kr_stock", uni)
    log.info("KR stocks: %d — downloading prices", len(uni))
    kdf = data.fetch_kr(uni["ticker"].tolist(), market="kr")
    frames = data.to_ticker_frames(kdf)
    name_map = dict(zip(uni["ticker"], uni["name"]))
    mcap_map = dict(zip(uni["ticker"], uni["market_cap_usd"]))
    return _finish_market("KR", frames, {
        "country": "KR", "asset_map": {t: "stock" for t in frames},
        "name_map": name_map, "sector_map": {}, "industry_map": {},
        "category_map": {}, "mcap_map": mcap_map,
        "benchmark": "^KS11", "market_s": regimes["KR"],
        "rs_asof": str(dt.date.today()),
    }, built)


def _finish_market(market, frames, meta_info, built):
    res = build.build_market(market, frames, meta_info, write_detail=True)
    built[market] = res
    log.info("%s: %d rows, %d signals", market, len(res["rows"]), res["grand_total"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr", "all"], default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="소량 티커 end-to-end (smoke_test.py로 위임)")
    args = ap.parse_args(argv)

    if args.smoke:
        from . import smoke
        return smoke.run()

    now_iso = _now_iso()
    built, regimes, fresh = {}, {}, {}
    updated = []

    # 벤치마크 → 시장 레짐
    try:
        benches = _bench_frames()
        regimes["US"] = build.market_regime(benches["^GSPC"])
        regimes["KR"] = build.market_regime(benches["^KS11"])
    except Exception as e:
        log.warning("benchmark fetch failed: %s", e)
        regimes = {"US": True, "KR": True}

    do_us = args.market in ("us", "all")
    do_kr = args.market in ("kr", "all")

    fresh = {"price_us": do_us, "price_kr": do_kr, "rs_us": do_us, "rs_kr": do_kr,
             "ma_signals": True, "box_stocks": True, "nhigh_stats": True,
             "nhigh_signals": True, "price_etf_us": do_us, "etf_bt_us": do_us}

    if do_us:
        try:
            run_us(now_iso, built, regimes, fresh)
            updated.append("US")
        except Exception as e:
            log.error("US build failed, keeping previous JSON: %s", e)
            fresh["rs_us"] = False

    if do_kr:
        try:
            rate = data.usdkrw()
            run_kr(now_iso, built, regimes, fresh, rate)
            updated.append("KR")
        except Exception as e:
            log.error("KR build failed, keeping previous JSON: %s", e)
            fresh["rs_kr"] = False

    out = build.merge_and_write(built, updated, regimes, fresh, now_iso,
                                write_detail=True)
    log.info("wrote buy-signals.json: %d rows (updated: %s)",
             out["count"], updated or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
