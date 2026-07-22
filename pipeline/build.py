"""메인 배치 오케스트레이터 — 유니버스 → 신호 → 백테스트 → 대표신호 → JSON.

명세서 §2.1(meta/rows), §2.3(자격), §2.4(최적기간), §2.5(zone), §2.6(별점) 구현.

시장별 부분 갱신을 지원한다: build_market()이 시장 하나의 rows/detail을
계산하고, merge_and_write()가 기존 JSON과 병합해 meta.warnings에 지연을 기록.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from . import backtest, config, data, detail, phrase, signals, stars
from .rs import rs_percentiles, rs_raw

log = logging.getLogger(__name__)

FILT_LABEL = {"L": "장기", "S": "단기"}


# ── 시장 레짐 ([자체 기준] 6)

def market_regime(bench_df: pd.DataFrame) -> bool:
    closes = bench_df["close"]
    if len(closes) < config.REGIME_SMA:
        return False
    sma = closes.rolling(config.REGIME_SMA).mean().iloc[-1]
    return bool(closes.iloc[-1] > sma)


# ── 한 티커의 모든 신호 후보 산출

def _ma_candidates(df: pd.DataFrame, closes: np.ndarray) -> list[dict]:
    out = []
    scan = signals.ma_signal_scan(df["close"].reset_index(drop=True))
    for key, s in scan.items():
        if not s["current"] or s["ma_value"] is None:
            continue
        filt, ma_type, period = key.split("|")
        period_rows = backtest.backtest_signal(closes, s["events"], config.HOLDS_MA)
        best = backtest.select_optimal(period_rows)
        if best is None:
            continue
        out.append({
            "signal": "이평", "detail": f"{filt}/{ma_type}{period}",
            "ref_price": s["ma_value"], "best": best,
            "all_periods": len(period_rows),
        })
    return out


def _box_candidates(df: pd.DataFrame, closes: np.ndarray) -> list[dict]:
    out = []
    scan = signals.box_signal_scan(df["high"].reset_index(drop=True),
                                   df["low"].reset_index(drop=True),
                                   df["close"].reset_index(drop=True))
    for filt, s in scan.items():
        if not s["current"]:
            continue
        best = backtest.select_optimal(
            backtest.backtest_signal(closes, s["events"], config.HOLDS_BOX))
        if best is None:
            continue
        out.append({"signal": "박스", "detail": filt,
                    "ref_price": s["ref_price"], "best": best})
    return out


def _nhigh_candidates(df: pd.DataFrame, closes: np.ndarray) -> list[dict]:
    out = []
    scan = signals.nhigh_signal_scan(df["close"].reset_index(drop=True))
    for w, s in scan.items():
        if not s["current"]:
            continue
        best = backtest.select_optimal(
            backtest.backtest_signal(closes, s["events"], config.HOLDS_NHIGH))
        if best is None:
            continue
        out.append({"signal": "신고가", "detail": w,
                    "ref_price": s["ref_price"], "best": best})
    return out


def ticker_signals(df: pd.DataFrame) -> list[dict]:
    """현재 신호가 살아있고 최적기간(표본≥20, 손실有)이 잡히는 모든 후보."""
    closes = df["close"].to_numpy()
    return (_ma_candidates(df, closes) + _box_candidates(df, closes)
            + _nhigh_candidates(df, closes))


# ── 별점 태그 계산

def compute_star(df: pd.DataFrame, rs: int, market_cap_usd, asset: str,
                 vol_q4_cut: float | None):
    closes = df["close"].to_numpy()
    vol = stars.volatility_60d(closes)
    wick = stars.upper_wick_days(df["open"].to_numpy(), df["high"].to_numpy(),
                                 df["close"].to_numpy())
    surged = stars.surge_flags(closes)
    tags = stars.compute_tags(vol=vol, vol_q4_cut=vol_q4_cut, wick_days=wick,
                              rs=rs, market_cap_usd=market_cap_usd, surged=surged)
    return stars.stars_from_tags(tags, asset)


# ── 시장 하나 빌드

def build_market(market: str, frames: dict[str, pd.DataFrame], meta_info: dict,
                 write_detail: bool = True) -> dict:
    """market: "US" | "KR". meta_info: {country, asset_map, name_map, sector_map,
    industry_map, category_map, mcap_map, benchmark, market_s}.

    반환: {"rows": [...], "grand_total": int, "star_counts": {...},
    "rs_asof": str, "detail": {ticker: {ma/box/nhigh json}}}.
    """
    # 1) RS 백분위 (시장 내)
    raw = {t: rs_raw(df["close"].to_numpy()) for t, df in frames.items()}
    raw = {t: v for t, v in raw.items() if v is not None}
    rs_pct = rs_percentiles(raw)

    # 2) 변동성 Q4 컷 (유니버스 상위 25%)
    vols = {t: stars.volatility_60d(df["close"].to_numpy()) for t, df in frames.items()}
    vols_valid = [v for v in vols.values() if v is not None]
    vol_q4_cut = float(np.quantile(vols_valid, 0.75)) if vols_valid else None

    # 2b) 섹터·산업 그룹 평균 RS (§2.7 카드 펼침 칩)
    group_rs = _group_rs(frames, rs_pct, meta_info)

    rows, grand_total = [], 0
    star_counts = {"★★★": 0, "★★": 0, "★": 0}
    detail_out = {}
    asset_map = meta_info["asset_map"]

    for t, df in frames.items():
        rs = rs_pct.get(t)
        if rs is None or rs < config.RS_MIN:
            continue
        cands = ticker_signals(df)
        if not cands:
            continue
        grand_total += len(cands)

        asset = asset_map.get(t, "stock")
        # 대표 신호 = EV 최대
        rep = max(cands, key=lambda c: c["best"]["ev"])
        b = rep["best"]
        cur_price = float(df["close"].iloc[-1])
        from .zone import buy_zone, in_zone
        if not in_zone(rep["signal"], rep["ref_price"], cur_price):
            continue
        zlow, zhigh = buy_zone(rep["signal"], rep["ref_price"])

        mcap = meta_info["mcap_map"].get(t)
        s_stars, cut_reason = compute_star(df, rs, mcap, asset, vol_q4_cut)
        if s_stars == 3:
            star_counts["★★★"] += 1
        elif s_stars == 2:
            star_counts["★★"] += 1
        elif s_stars == 1:
            star_counts["★"] += 1

        detail_str = rep["detail"]
        rows.append({
            "ticker": t,
            "name": meta_info["name_map"].get(t) if asset == "stock" else None,
            "country": meta_info["country"],
            "asset": asset,
            "signal": rep["signal"],
            "detail": detail_str,
            "hold_period": b["hold"],
            "ev": round(b["ev"], 4),
            "win_rate": b["win_rate"],
            "avg_win": b["avg_win"],
            "avg_loss": b["avg_loss"],
            "pl_ratio": b["pl_ratio"],
            "n": b["n"],
            "rs": rs,
            "phrase": phrase.make_phrase(rep["signal"], detail_str, b["hold"]),
            "category": meta_info["category_map"].get(t) if asset == "etf" else None,
            "sector": meta_info["sector_map"].get(t),
            "industry": meta_info["industry_map"].get(t),
            "naver_sector": None, "naver_theme": None,
            "market_cap_usd": mcap,
            "ref_price": round(rep["ref_price"], 4),
            "cur_price": round(cur_price, 4),
            "zone_low": round(zlow, 4),
            "zone_high": round(zhigh, 4),
            "stars": s_stars,
            "star_score": None,
            "cut_reason": cut_reason,
            "lottery_flag": 0,
        })

        if write_detail:
            bench = meta_info["benchmark"]
            detail_out[t] = {
                "ma": detail.ma_by_ticker(t, df, bench, meta_info["market_s"]),
                "box": detail.box_by_ticker(t, df),
                "nhigh": detail.nhigh_by_ticker(t, df, meta_info["country"]),
            }

    return {
        "rows": rows,
        "grand_total": grand_total,
        "star_counts": star_counts,
        "stock_n": sum(1 for r in rows if r["asset"] == "stock"),
        "etf_n": sum(1 for r in rows if r["asset"] == "etf"),
        "rs_asof": meta_info["rs_asof"],
        "detail": detail_out,
        "group_rs": group_rs,
    }


def _group_rs(frames, rs_pct, meta_info) -> dict:
    """섹터/산업별 평균 RS (표본수 포함). 반환: {"sectors": {...}, "industries": {...}}."""
    def agg(key_map):
        buckets = {}
        for t in frames:
            g = key_map.get(t)
            r = rs_pct.get(t)
            if not g or r is None:
                continue
            buckets.setdefault(g, []).append(r)
        return {g: {"avg_rs": round(float(np.mean(v)), 1), "count": len(v)}
                for g, v in buckets.items()}
    return {"sectors": agg(meta_info.get("sector_map", {})),
            "industries": agg(meta_info.get("industry_map", {}))}


# ── 정렬 (명세서 §2.7: 별 우선 = stars desc, null 최후미 → ev desc)

def sort_stars_first(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (-(r["stars"] if r["stars"] is not None else -1),
                                       -r["ev"]))


def sort_ev(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: -r["ev"])


# ── 시장 결과 병합 + JSON 쓰기

def _empty_meta(now_iso: str) -> dict:
    return {
        "built_at": now_iso,
        "market_s": {},
        "materials_fresh": {},
        "warnings": [],
        "rs_min": config.RS_MIN,
        "min_sample": config.MIN_SAMPLE,
        "by_country": {"KR": 0, "US": 0},
        "by_asset": {"stock": 0, "etf": 0},
        "grand_total": 0,
        "star": {"KR": {}, "US": {},
                 "rs_source": "rs_scores.pct (live)",
                 "rs_asof": {}, "star_cut": {"KR": 4}},
    }


def load_existing() -> dict | None:
    p = config.SITE_DATA_DIR / "buy-signals.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def merge_and_write(built: dict[str, dict], updated_markets: list[str],
                    regimes: dict[str, bool], fresh: dict[str, bool],
                    now_iso: str, write_detail: bool = True) -> dict:
    """built: {"US": build_market결과, ...}. updated_markets: 이번에 갱신한 시장.

    갱신하지 않은 시장은 기존 JSON rows를 그대로 유지(직전 데이터 보존).
    """
    existing = load_existing()
    meta = existing["meta"] if existing else _empty_meta(now_iso)
    meta["built_at"] = now_iso
    meta["market_s"] = {config.BENCHMARKS[m]: regimes.get(m, False)
                        for m in ("KR", "US")}
    meta["materials_fresh"] = fresh
    meta["warnings"] = []

    # rows: 갱신 시장은 새 값, 나머지는 기존값 유지
    prev_rows = existing["rows"] if existing else []
    kept = [r for r in prev_rows if r["country"] not in updated_markets]
    new_rows = []
    for m in updated_markets:
        new_rows.extend(built[m]["rows"])
    all_rows = kept + new_rows
    all_rows = sort_ev(all_rows)  # 명세서: 도착 시 ev 내림차순

    # meta 집계
    by_country = {"KR": 0, "US": 0}
    by_asset = {"stock": 0, "etf": 0}
    for r in all_rows:
        by_country[r["country"]] = by_country.get(r["country"], 0) + 1
        by_asset[r["asset"]] = by_asset.get(r["asset"], 0) + 1
    meta["by_country"] = by_country
    meta["by_asset"] = by_asset

    grand = sum(built[m]["grand_total"] for m in updated_markets)
    grand += (meta.get("grand_total", 0) if not updated_markets else 0)
    meta["grand_total"] = grand if updated_markets else meta.get("grand_total", 0)

    # star 블록
    for m in ("KR", "US"):
        if m in updated_markets:
            b = built[m]
            meta["star"][m] = {
                "stock_n": b["stock_n"], "★★★": b["star_counts"]["★★★"],
                "★★": b["star_counts"]["★★"], "★": b["star_counts"]["★"],
                "컷": 0, "lottery": 0, "ax3_unmeasured": 0, "ax6_nodata": 0,
            }
            meta["star"]["rs_asof"][m] = b["rs_asof"]
        else:
            meta["star"].setdefault(m, {})

    # 지연 경고 ([명세서] "KR RS 지연" 재현)
    for m in ("KR", "US"):
        rs_key = f"rs_{m.lower()}"
        if not fresh.get(rs_key, True):
            label = {"KR": "KR RS 지연 — 신호가 묵었을 수 있음",
                     "US": "US RS 지연 — 신호가 묵었을 수 있음"}[m]
            meta["warnings"].append(label)
        if not regimes.get(m, True):
            meta["warnings"].append(
                f"{config.BENCHMARKS[m]} 시장 레짐 하락(약세) — 신호 신뢰도 주의")

    out = {"meta": meta, "count": len(all_rows), "total": len(all_rows),
           "page": 1, "limit": 1000, "sort": "ev", "rows": all_rows}

    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "buy-signals.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=None))

    if write_detail:
        config.DETAIL_DIR.mkdir(parents=True, exist_ok=True)
        for m in updated_markets:
            for t, d in built[m]["detail"].items():
                safe = t.replace("/", "_")
                (config.DETAIL_DIR / f"{safe}.json").write_text(
                    json.dumps(d, ensure_ascii=False))

    # 섹터·산업 그룹 RS (§2.7 칩) — 시장별 파일
    for m in updated_markets:
        gr = built[m].get("group_rs")
        if not gr:
            continue
        gdir = config.SITE_DATA_DIR / "rs" / m.lower()
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "sectors.json").write_text(
            json.dumps(gr["sectors"], ensure_ascii=False))
        (gdir / "industries.json").write_text(
            json.dumps(gr["industries"], ensure_ascii=False))
    return out
