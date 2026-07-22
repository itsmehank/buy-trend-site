"""스모크 테스트 — 소량 티커로 파이프라인 전체를 end-to-end 실행하고
JSON 스키마가 명세서 §2.1과 일치하는지 검증한다.

네트워크(yfinance) 접근이 필요하다. 실행:  python -m pipeline.run --smoke
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from . import build, config, data, trading

log = logging.getLogger("smoke")

# 미국 주식 8 + ETF 2 (추세·박스·신고가가 두루 나오도록 대형 성장주 위주)
SMOKE_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TRGP"]
SMOKE_ETFS = ["SPY", "QQQ"]

# rows[i] 필수 필드 (명세서 §2.1)
REQUIRED_ROW_FIELDS = [
    "ticker", "name", "country", "asset", "signal", "detail", "hold_period",
    "ev", "win_rate", "avg_win", "avg_loss", "pl_ratio", "n", "rs", "phrase",
    "category", "sector", "industry", "naver_sector", "naver_theme",
    "market_cap_usd", "ref_price", "cur_price", "zone_low", "zone_high",
    "stars", "star_score", "cut_reason", "lottery_flag",
]
REQUIRED_META_FIELDS = [
    "built_at", "market_s", "materials_fresh", "warnings", "rs_min",
    "min_sample", "by_country", "by_asset", "grand_total", "star",
]


def _validate(out: dict) -> list[str]:
    errs = []
    meta = out.get("meta", {})
    for f in REQUIRED_META_FIELDS:
        if f not in meta:
            errs.append(f"meta 필드 누락: {f}")
    if meta.get("rs_min") != config.RS_MIN:
        errs.append(f"rs_min != {config.RS_MIN}")
    if meta.get("min_sample") != config.MIN_SAMPLE:
        errs.append(f"min_sample != {config.MIN_SAMPLE}")
    for key in ("count", "total", "page", "limit", "sort", "rows"):
        if key not in out:
            errs.append(f"top-level 필드 누락: {key}")
    if out.get("sort") != "ev":
        errs.append("sort != 'ev'")

    rows = out.get("rows", [])
    # ev 내림차순 확인
    evs = [r["ev"] for r in rows]
    if evs != sorted(evs, reverse=True):
        errs.append("rows가 ev 내림차순이 아님")

    for r in rows:
        for f in REQUIRED_ROW_FIELDS:
            if f not in r:
                errs.append(f"{r.get('ticker','?')}: row 필드 누락 {f}")
        # zone 내 포함 (§2.3)
        if not (r["zone_low"] <= r["cur_price"] <= r["zone_high"]):
            errs.append(f"{r['ticker']}: cur_price가 zone 밖")
        # RS 자격 (§2.3)
        if r["rs"] < config.RS_MIN:
            errs.append(f"{r['ticker']}: rs < {config.RS_MIN}")
        # 표본 자격 (§2.3)
        if r["n"] < config.MIN_SAMPLE:
            errs.append(f"{r['ticker']}: n < {config.MIN_SAMPLE}")
        # 신호 종류
        if r["signal"] not in ("이평", "박스", "신고가"):
            errs.append(f"{r['ticker']}: 알 수 없는 signal {r['signal']}")
        # ETF는 stars null, 주식은 1~3
        if r["asset"] == "etf" and r["stars"] is not None:
            errs.append(f"{r['ticker']}: ETF인데 stars != null")
        if r["asset"] == "stock" and r["stars"] not in (1, 2, 3):
            errs.append(f"{r['ticker']}: 주식 stars 범위 오류 {r['stars']}")
    return errs


def run() -> int:
    log.info("스모크: %d 주식 + %d ETF 다운로드", len(SMOKE_STOCKS), len(SMOKE_ETFS))
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    benches = data.fetch_benchmarks()
    regimes = {"US": build.market_regime(benches["^GSPC"]),
               "KR": build.market_regime(benches["^KS11"])}

    sdf = data.fetch_us(SMOKE_STOCKS, market="smoke_stock")
    edf = data.fetch_us(SMOKE_ETFS, market="smoke_etf")
    s_frames = data.to_ticker_frames(sdf)
    e_frames = data.to_ticker_frames(edf)
    frames = {**s_frames, **e_frames}
    asset_map = {**{t: "stock" for t in s_frames}, **{t: "etf" for t in e_frames}}

    meta_info = {
        "country": "US", "asset_map": asset_map,
        "name_map": {t: t for t in frames},
        "sector_map": {t: "Technology" for t in s_frames},
        "industry_map": {t: "Software" for t in s_frames},
        "category_map": {}, "mcap_map": {t: 1e12 for t in s_frames},
        "benchmark": "^GSPC", "market_s": regimes["US"],
        "rs_asof": str(trading.latest_complete_date("US")),
    }
    built = {"US": build.build_market("US", frames, meta_info, write_detail=True)}
    fresh = {"price_us": True, "rs_us": True, "price_etf_us": True, "etf_bt_us": True,
             "ma_signals": True, "box_stocks": True, "nhigh_stats": True,
             "nhigh_signals": True}
    out = build.merge_and_write(built, ["US"], regimes, fresh, now_iso,
                                write_detail=True)

    errs = _validate(out)
    log.info("생성된 rows: %d, grand_total: %d", out["count"], out["meta"]["grand_total"])
    if errs:
        log.error("스키마 검증 실패 %d건:", len(errs))
        for e in errs[:30]:
            log.error("  - %s", e)
        return 1
    log.info("✅ 스키마 검증 통과 — buy-signals.json이 명세서 §2.1과 일치")
    # 상세 파일 하나 확인
    if out["rows"]:
        t = out["rows"][0]["ticker"]
        dpath = config.DETAIL_DIR / f"{t.replace('/', '_')}.json"
        if dpath.exists():
            d = json.loads(dpath.read_text())
            assert {"ma", "box", "nhigh"} <= set(d), "detail 3종 누락"
            log.info("✅ detail/%s.json 3종(ma/box/nhigh) 생성 확인", t)
    return 0
