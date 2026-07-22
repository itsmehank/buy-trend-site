# 매수 · 추세추종 — 매일 자동 갱신되는 정적 매수 사이트

`buy-tab-spec.md`(easyinvesting.app「매수」탭 리버스 엔지니어링 명세서)의 **추세추종
탭**을 그대로 재현한, 매일 자동 갱신되는 정적 사이트다. 서버 없이 GitHub Actions 배치가
JSON을 생성하고 GitHub Pages가 서빙한다.

- **파이프라인** (`pipeline/`): 미국 주식·ETF(yfinance) + 한국 주식(pykrx 수정주가)의
  상대강도(RS)·기술적 신호(이평 눌림·박스 돌파·신고가)를 검출하고 8기간 백테스트로
  최적 보유기간과 기대값(EV)을 산출한다.
- **프론트엔드** (`site/`): 빌드 도구 없는 순수 HTML/CSS/JS. `site/data/buy-signals.json`
  하나만 받아 필터·정렬·상위 100종목 카드를 그리고, 카드 펼침 시에만 종목별 상세 JSON을
  lazy fetch 한다.
- **자동화** (`.github/workflows/daily.yml`): 미국·한국 장 마감 후 각 1회 실행, 해당 시장만
  갱신해 JSON을 병합, Pages로 배포.

명세서의 ✅(검증된 사실) 항목은 그대로 따랐고, ❓(서버 내부 파라미터) 항목은 아래
[자체 기준]으로 결정했다. 따라서 **구조·UX는 동일하되 개별 종목 선정 결과는 원본과
다를 수 있다.**

---

## 화면 구성 (명세서 §2.7 재현)

- **상단 배너**: 시장(미국/한국)별 데이터 기준일 + 최신/장중/지연 상태 칩
- **필터 칩 4종**: 국가(전체/한국/미국) · 자산(전체/주식/ETF) · 신호(전체/이평/박스/신고가)
  · 시총(전체/≥$1B/≥$10B/≥$100B/≥$1T)
- **정렬 4종**: 별 우선(기본) / EV순 / 승률순 / RS순
- **카드(접힘)**: 순번, 별점, 종목명, 신호 배지(이평 파랑·박스 초록·신고가 주황), phrase
- **카드(펼침)**: 섹터·산업 RS 칩, 매수 구간 + `구간 내` 배지 + 종가, 지표 6개,
  8기간 성적표(최적 보유기간 노란 하이라이트 ★, 손실 없는 기간 "—")
- 한국 종목은 원화(₩), 미국 종목은 달러($) 표기

---

## 로컬 실행

### 1) 환경 준비

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 단위 테스트 (수치 로직 검증)

```bash
python -m pytest -q
```

EV 공식, 최적 보유기간 선정(무손실 기간 제외·표본 미달 제외), 매수 구간, 별점,
터치·돌파 정의, RS, phrase를 모두 검증한다. (명세서 §2.4 검증 사례 포함)

### 3) 스모크 테스트 (소량 티커 end-to-end)

```bash
python -m pipeline.run --smoke
```

미국 주식 8 + ETF 2를 실제로 내려받아 파이프라인 전체를 돌리고, 생성된
`site/data/buy-signals.json`이 명세서 §2.1 스키마와 일치하는지 검증한다.
(네트워크 필요)

### 4) 프론트 확인

```bash
cd site
python -m http.server 8000
# 브라우저에서 http://localhost:8000
```

프론트는 `site/data/` 아래 정적 JSON만 읽으므로 별도 서버가 필요 없다.

### 5) 전체 유니버스 배치 (선택)

```bash
python -m pipeline.run --market us     # 미국 주식+ETF
python -m pipeline.run --market kr     # 한국 주식
python -m pipeline.run --market all    # 둘 다
```

> ⚠️ 첫 실행은 유니버스 전체(수천 종목 × 15년)를 내려받아 시간이 오래 걸린다.
> 시간이 부담되면 `pipeline/config.py`의 시총 하한(`US_MIN_MCAP_USD`,
> `KR_MIN_MCAP_KRW`)을 올려 종목 수를 줄이면 된다. 두 번째 실행부터는 parquet
> 증분 캐시로 당일 데이터만 받는다.

---

## GitHub 배포 (Secrets 불필요)

이 사이트는 공개 데이터 소스(yfinance/pykrx/NASDAQ 스크리너)만 쓰므로 **API 키·Secrets가
전혀 필요 없다.** `GITHUB_TOKEN`(자동 제공)만으로 JSON 커밋·Pages 배포가 된다.

### 단계별 절차

1. **저장소 생성 후 push**
   ```bash
   git init && git add . && git commit -m "init: 매수 추세추종 사이트"
   git branch -M main
   git remote add origin https://github.com/{계정명}/{저장소명}.git
   git push -u origin main
   ```

2. **Pages 활성화**
   저장소 **Settings → Pages → Build and deployment → Source**를 **"GitHub Actions"**로
   설정한다. (브랜치 방식이 아니라 Actions 방식)

3. **Secrets 설정** — 미국(US)·ETF는 **Secrets 없이** 동작한다. 다만 **한국(KR)**은
   2026-02-27 KRX API 포맷 변경 이후 스냅샷 API(유니버스·시가총액)에 **로그인이
   필수**가 됐다(pykrx [이슈 #276](https://github.com/sharebook-kr/pykrx/issues/276)).
   KR을 갱신하려면 Settings → Secrets and variables → Actions → New repository secret 로
   아래 2개를 추가한다:
   - `KRX_ID` — data.krx.co.kr 로그인 ID
   - `KRX_PW` — data.krx.co.kr 로그인 비밀번호

   Secrets가 없으면 KR은 자동으로 건너뛰고(US만 갱신) 배너에 "KR 갱신 건너뜀"이 뜬다.
   워크플로 권한은 `daily.yml`의 `permissions:` 블록에 선언돼 있다. push가 권한 오류로
   실패하면 Settings → Actions → General → Workflow permissions 를
   **"Read and write permissions"**로 설정한다.

   > 로컬 실행 시에는 저장소 루트에 `.env`(gitignore됨)를 만들어 `KRX_ID=...`,
   > `KRX_PW=...` 두 줄을 넣으면 파이프라인이 자동으로 로그인한다.

4. **첫 배치 수동 실행**
   **Actions 탭 → daily-buy-signals → Run workflow** 로 `workflow_dispatch`를 실행한다.
   시장을 `all`(또는 `us`)로 선택. 이 실행이 첫 `buy-signals.json`을 생성·커밋하고
   Pages를 배포한다.

5. **접속 확인**
   `https://{계정명}.github.io/{저장소명}/`

이후에는 스케줄(미국 07:00 KST, 한국 16:30 KST)로 자동 갱신된다. 각 실행은 해당 시장
데이터만 갱신하고 다른 시장의 직전 JSON은 그대로 유지하며, 실패해도 마지막 성공 JSON이
남아 사이트가 깨지지 않는다.

---

## [자체 기준] — 명세서 ❓ 항목 결정 (`pipeline/config.py`)

| # | 항목 | 결정 |
|---|---|---|
| 1 | RS 산식 | IBD식: 0.4×3M + 0.2×6M + 0.2×9M + 0.2×12M 수익률 → 시장별(미국/한국) 백분위 0~100. 벤치마크 ^GSPC / ^KS11 |
| 2 | 이평 근접 | 종가가 이평선 −2%~+3% 이내. 장기(L)=종가>SMA200 & SMA50>SMA200, 단기(S)=종가>SMA50 |
| 3 | 박스 정의 | 최근 20일↑ 고저 진폭 ≤15% 횡보 상단 종가 돌파. 장기=60일↑, 단기=20일↑. 재돌파는 박스 재형성 후만 |
| 4 | 터치 1회 | 근접 상태 진입일을 1회로 카운트, 벗어났다 재진입해야 새 터치 |
| 5 | 변동성 Q4 | 60일 일간수익률 표준편차 유니버스 상위 25%. 윗꼬리 5일+ = 최근 10일 중 윗꼬리>몸통 5일↑ |
| 6 | 시장 레짐 | 벤치마크 종가 > SMA200 이면 true. false면 `meta.warnings`에 경고 |
| 7 | 유니버스 | 미국: 시총 ≥$300M & 20일 평균 거래대금 ≥$2M / 한국: KOSPI+KOSDAQ 시총 ≥3000억 / ETF: 20일 거래대금 상위 500(AUM 프록시). 이력 최대 15년 |

> ETF는 AUM 공개 API가 없어 20일 평균 거래대금을 AUM 근사 프록시로 사용한다.

---

## 산출 데이터 스키마 (명세서 §2.1)

- `site/data/buy-signals.json` — `{ meta, count, total, page, limit, sort, rows }`.
  `rows`는 티커당 대표 신호 1개(EV 최대), EV 내림차순 정렬.
- `site/data/detail/{ticker}.json` — `{ ma, box, nhigh }` 3종(§2.7의 by-ticker 응답).
- `site/data/rs/{us|kr}/{sectors,industries}.json` — 섹터·산업 평균 RS(카드 펼침 칩).

핵심 수식은 모두 단위 테스트로 고정돼 있다(`tests/test_core.py`).

## 디렉터리 구조

```
pipeline/       배치 파이프라인 (Python 3.11)
  config.py     파라미터(명세서 ✅ + [자체 기준])
  data.py       가격 수집 + parquet 증분 캐시 (yfinance / pykrx)
  universe.py   유니버스 구성
  rs.py         IBD식 상대강도
  signals.py    신호 3종 검출
  backtest.py   8기간 백테스트 · EV · 최적 보유기간
  zone.py       매수 구간
  stars.py      별점(감점 태그)
  phrase.py     카드 문구
  detail.py     by-ticker 상세 3종
  build.py      오케스트레이터 · 시장 병합 · JSON 생성
  run.py        CLI 진입점
  smoke.py      end-to-end 스모크 + 스키마 검증
site/           정적 프론트엔드 (HTML/CSS/JS)
tests/          pytest 단위 테스트
.github/workflows/daily.yml   일간 자동 배치
```
