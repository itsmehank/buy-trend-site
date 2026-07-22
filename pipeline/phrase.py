"""카드 배지 문구(phrase) 생성 규칙 (명세서 §2.2 ✅)."""
from __future__ import annotations

NHIGH_LABEL = {"ATH": "역대최고", "20": "20일신고가", "55": "55일신고가"}
FILT_LABEL = {"L": "장기", "S": "단기"}


def make_phrase(signal: str, detail: str, hold_period: int) -> str:
    if signal == "이평":
        filt, ma_type, period = detail.split("/")[0], *_split_ma(detail)
        return f"{ma_type}{period} {FILT_LABEL[filt]} 근접 · {hold_period}일 보유"
    if signal == "박스":
        return f"박스돌파 {FILT_LABEL[detail]} · {hold_period}일 보유"
    if signal == "신고가":
        return f"{NHIGH_LABEL[detail]} 돌파 · {hold_period}일 보유"
    raise ValueError(f"unknown signal: {signal}")


def _split_ma(detail: str) -> tuple[str, str]:
    """"L/SMA10" → ("SMA", "10")."""
    body = detail.split("/")[1]
    ma_type = "SMA" if body.startswith("SMA") else "EMA"
    return ma_type, body[len(ma_type):]


def nhigh_table_title(detail: str) -> str:
    label = {"ATH": "역대최고(ATH)", "20": "20일신고가", "55": "55일신고가"}[detail]
    return f"{label} · 신고가 8기간 성적표"
