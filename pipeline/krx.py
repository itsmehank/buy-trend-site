"""KRX 접근 부트스트랩 — pykrx 로그인 + hang 방지 타임아웃 가드.

2026-02-27 KRX API 포맷 변경 이후 스냅샷 계열(get_market_cap·get_market_ticker_list
등)은 로그인이 필수가 됐다. 이 pykrx 빌드는 import 시 KRX_ID/KRX_PW 환경변수로
자동 로그인한다. 따라서 이 모듈을 **pykrx import 전에** import해서
(1) 타임아웃 가드를 걸고 (2) 자격증명이 있는지 미리 확인한다.

로컬 실행 편의를 위해 저장소 루트의 .env(있으면)에서 KRX_ID/KRX_PW를 읽어온다.
CI에서는 GitHub Secrets를 env로 주입한다(자격증명을 코드/저장소에 넣지 않는다).
"""
from __future__ import annotations

import logging
import os
import socket

import requests

from .config import ROOT

log = logging.getLogger(__name__)

# ── (1) hang 방지: pykrx는 timeout을 안 걸어 KRX throttle 시 소켓에서 무한 대기.
#     socket 기본 타임아웃 + requests 어댑터에 기본 (connect, read) 타임아웃 주입.
#     (kr-by-claude의 2026-06 daily-chain hang 사고 대응과 동일한 방식)
_SOCKET_TIMEOUT = 30
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
socket.setdefaulttimeout(_SOCKET_TIMEOUT)

_orig_send = requests.adapters.HTTPAdapter.send


def _send_with_timeout(self, request, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
    return _orig_send(self, request, **kwargs)


requests.adapters.HTTPAdapter.send = _send_with_timeout


# ── (2) 자격증명 로드: 저장소 루트 .env → 없으면 이미 설정된 env 사용.
def _load_dotenv() -> None:
    envfile = ROOT / ".env"
    if not envfile.exists():
        return
    for line in envfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k in ("KRX_ID", "KRX_PW") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def has_credentials() -> bool:
    _load_dotenv()
    return bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))


def ensure_login() -> bool:
    """KR 페치 전에 호출. 자격증명이 있으면 pykrx가 import 시 로그인한다.

    반환: 로그인 자격증명 사용 가능 여부. 없으면 False(호출자가 KR을 건너뛰도록).
    """
    if not has_credentials():
        log.warning("KRX_ID/KRX_PW 미설정 — KR 스냅샷 API(유니버스·시총)는 "
                    "로그인 없이는 빈 응답을 반환한다. KR 갱신을 건너뛴다.")
        return False
    # 자격증명이 env에 있으면, 이 시점 이후의 첫 pykrx import에서 자동 로그인된다.
    return True
