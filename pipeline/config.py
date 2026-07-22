"""파이프라인 전역 설정 — 명세서 ✅ 값과 [자체 기준] 파라미터."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
SITE_DATA_DIR = ROOT / "site" / "data"
DETAIL_DIR = SITE_DATA_DIR / "detail"

# ── 1차 자격 (명세서 §2.3 ✅)
RS_MIN = 90
MIN_SAMPLE = 20

# ── 보유기간 그리드 (명세서 §2.4 ✅ — 박스만 63일 사용)
HOLDS_MA = [2, 3, 5, 10, 20, 60, 126, 252]
HOLDS_NHIGH = [2, 3, 5, 10, 20, 60, 126, 252]
HOLDS_BOX = [2, 3, 5, 10, 20, 63, 126, 252]

# ── 이평 세트 (명세서 §2.2 ✅ — EMA만 20 대신 21)
SMA_PERIODS = [10, 20, 30, 40, 50, 60]
EMA_PERIODS = [10, 21, 30, 40, 50, 60]

# ── 매수 구간 (명세서 §2.5 ✅)
ZONE_MA_LOW = 0.98
ZONE_MA_HIGH = 1.03
ZONE_BREAKOUT_HIGH = 1.05

# ── [자체 기준] 2: 이평 근접 판정 = 종가가 이평선의 −2%~+3% 이내
MA_NEAR_LOW = 0.98
MA_NEAR_HIGH = 1.03

# ── [자체 기준] 3: 박스 정의
BOX_AMP_MAX = 0.15        # 고저 진폭 ≤ 15%
BOX_MIN_DAYS_L = 60       # 장기 박스 유지 기간
BOX_MIN_DAYS_S = 20       # 단기 박스 유지 기간

# ── 신고가 윈도우 (명세서 §2.2 ✅)
NHIGH_WINDOWS = ["ATH", 20, 55]

# ── 현재 신호로 인정하는 돌파 후 최대 경과일 (박스·신고가)
BREAKOUT_MAX_AGE = 5

# ── [자체 기준] 5: 감점 태그 파라미터
VOL_WINDOW = 60           # 변동성: 60일 일간수익률 표준편차
WICK_LOOKBACK = 10        # 최근 10거래일 중
WICK_MIN_DAYS = 5         # 윗꼬리 > 몸통인 날 5일 이상
RS_OVERHEAT = 97          # RS 97+ 초과열
# 대형주 무리한급등 (명세서 태그 문구에 파라미터 명시 ✅)
LARGE_CAP_USD = 10e9      # "시총대형" 기준 = $10B
SURGE_20D = 0.50          # 직전 20일 ≥ +50%
SURGE_1D_MAX = 0.20       # 60일 내 최대 일간 ≥ +20%

# ── [자체 기준] 1: RS 산식 (IBD식)
RS_WEIGHTS = {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2}
BENCHMARKS = {"US": "^GSPC", "KR": "^KS11"}

# ── [자체 기준] 6: 시장 레짐 = 벤치마크 종가 > SMA200
REGIME_SMA = 200

# ── [자체 기준] 7: 유니버스 범위
US_MIN_MCAP_USD = 300e6
US_MIN_DOLLAR_VOL = 2e6   # 20일 평균 거래대금
KR_MIN_MCAP_KRW = 300e9   # 3000억원
ETF_TOP_N = 500           # 미국 상장 AUM 상위 500
HISTORY_YEARS = 15

# ── 추세 맥락 (L/S) 판정용
TREND_LONG_SMA = 200      # L: 종가 > SMA200 & SMA50 > SMA200
TREND_SHORT_SMA = 50      # S: 종가 > SMA50
