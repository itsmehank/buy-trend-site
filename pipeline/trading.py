"""완결 거래일(as-of) 판정 — 장중 미완결 봉을 신호 계산에서 배제.

장이 열려 있는 동안 데이터 소스(pykrx·yfinance)는 '오늘' 봉을 실시간 중간값으로
반환한다. 이 미완결 봉으로 종가·이평·돌파를 계산하면 신호가 왜곡되므로, 각 시장의
'마지막으로 완결된 거래일'까지만 사용한다.

규칙: 시장 현지시간이 (장 마감 + 버퍼) 이후면 오늘 봉을 완결로 보고, 그 전이면
전일까지만 사용한다. 주말·공휴일은 애초에 오늘 봉이 없으므로 <= asof 트림으로
자연히 직전 거래일이 남는다(별도 휴장 캘린더 불필요).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

# 시장별 타임존과 '완결 간주' 기준 시각(장 마감 + 안전 버퍼)
MARKET_TZ = {"KR": ZoneInfo("Asia/Seoul"), "US": ZoneInfo("America/New_York")}
COMPLETE_AFTER = {
    "KR": dt.time(16, 0),   # KR 장 마감 15:30 + 30분 버퍼
    "US": dt.time(16, 15),  # US 장 마감 16:00 ET + 15분 버퍼
}


def latest_complete_date(market: str, now_utc: dt.datetime | None = None) -> dt.date:
    """market('KR'|'US')의 마지막 완결 거래일(date)."""
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    local = now_utc.astimezone(MARKET_TZ[market])
    d = local.date()
    if local.time() < COMPLETE_AFTER[market]:
        d = d - dt.timedelta(days=1)
    return d
