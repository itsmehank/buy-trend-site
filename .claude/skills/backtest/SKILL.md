---
name: backtest
description: Use when 이 저장소(buy-copy)에서 새 매수 전략 가설을 백테스트하거나, 테스트 요약을 조회하거나, 가설 풀을 감사할 때. "백테스트 돌려줘", "가설 테스트 해줘", "/backtest", "/backtest status", "/backtest audit" 같은 요청에 사용. 새 가설을 웹에서 수혈받는 작업은 backtest-refresh를 사용.
---

# backtest — 가설 백테스트 본 루프

가설 풀에서 하나를 꺼내 검토 게이트 → 테스트 → 검토 → 아카이브까지 수행한다.
**시작 전에 반드시 아래 3개 파일을 읽는다. 읽지 않고 진행 금지.**

1. `docs/analysis/PROTOCOL.md` — 공통 규약(데이터·고정 파라미터·체크리스트·운영 규칙)
2. `docs/analysis/hypotheses/registry.md` — 가설 대장
3. `docs/analysis/hypotheses/jail.md` — 테스트 불가 목록과 수감 기준

## 하드 가드 (위반 금지)

- 쓰기 허용 경로: `docs/analysis/**` 만. `pipeline/` `site/` `tests/` `scripts/` `.github/`는 읽기 전용
- 가격 데이터 재다운로드 금지 — 로컬 `cache/*.parquet`만 사용
- `git push` 금지 — 커밋만 하고 push는 사용자 판단
- 동시 실행 금지 — 다른 세션이 백테스트 중이면(작업 중인 미커밋 registry 변경 발견 시) 중단하고 사용자에게 알림
- 검토 sub-agent는 **Opus**(`model: opus`) 고정 — 3종((a)가설 게이트·(b)코드+결과·(c)문서) 모두. 세션 모델이 무엇이든 검토는 Opus로 띄운다 (근거: PROTOCOL §6)
- 검토는 **반드시 sub-agent로** 실행한다. 본 세션이 직접 검토하면 독립성이 없어 게이트가 무력해진다
- 본 세션이 Opus가 아니면 시작 시 "Opus 세션 권장"을 한 줄 안내만 하고 진행

## 인자 모드

| 인자 | 동작 |
|---|---|
| (없음) | 본 루프 실행 (아래 절차) |
| `status` | `docs/analysis/SUMMARY.md`를 읽어 그대로 출력 (테스트명 인자가 있으면 해당 항목만) |
| `audit` | 대기·후보 가설 전수를 jail 수감 기준과 대조 → 위반은 사유와 함께 감옥 이동, 결과 보고. 테스트는 하지 않음. 변경이 있으면 커밋 1개(push 금지), SUMMARY는 갱신하지 않음 |

## 본 루프 절차

### 1. 가설 선택
- registry `대기` 풀에서 맨 위 가설 1개를 선택한다. **직전 테스트와 다른 지표·다른 방식 우선** — 가설 키의 '진입' 지표가 registry 테스트 완료 최신 항목과 다른 것을 먼저 고른다.
- 선택 직전 그 가설을 jail 기준으로 자동 점검 — 테스트 불가면 감옥 이동 후 다음 가설.
- 풀이 비었으면: "가설 풀이 비었다. `/backtest-refresh`로 수혈하라"를 출력하고 종료.
- `btlib.loading.staleness_warning`으로 캐시 나이 확인, 6개월 초과면 경고를 기록해 둔다.

### 2. 가설 문서 초안 (결과 산출 전)
`docs/analysis/backtests/YYYY-MM-DD-<slug>.md` 초안 작성: 가설 키, 매수·매도 규칙의
일봉 종가 기준 명세, **사전 판정 기준**(PROTOCOL §3), 데이터 기준일, 난수 시드.

### 3. 가설 게이트 (최대 3회)
Agent 도구(`model: opus`)로 독립 검토를 돌린다. 프롬프트에 명시할 것:
- `docs/analysis/PROTOCOL.md` §4-(a) 체크리스트로 검토하라
- `registry.md` 전체·`jail.md`와 직접 대조하라 (파일 경로 전달)
- 반환: 통과/미통과 + 항목별 사유

미통과 → 사유 반영해 가설을 수정하고 재검토. **3회 미통과면 가설을 기각 처리**하고
1번으로 돌아가 다음 가설을 선택한다. 이때 registry 결과 요약에 **"게이트 미통과
(테스트 미실행)"**을 명기해 테스트 후 기각과 구분한다. (이 3회는
`/backtest-refresh`의 수혈 시도 3회와 별개 카운터다.)

### 4. 테스트 구현·실행
- 스크립트: `docs/analysis/backtests/scripts/<slug>.py` — **`btlib` 공용 모듈 사용**
  (`docs/analysis/btlib/`: loading·entries·exits·engine·metrics·regime·liquidity·costs).
  새 계산 로직이 필요하면 스크립트 안에 작성하되 문서에 이유를 남긴다.
- KR·US 양 시장 모두 실행 (PROTOCOL §3 — 한 시장 통과는 "부분 지지").
- 실행: 저장소 루트에서
  `PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/<slug>.py`
  (`PYTHONPATH`에 두 경로가 모두 있어야 `pipeline`과 `btlib`을 import할 수 있다)
- btlib를 수정했다면 `regression_check.py`를 다시 통과시켜야 한다.

### 5. 코드 + 결과 검토 (1회)
Agent(`model: opus`)에 스크립트 경로·실행 출력·PROTOCOL §4-(b) 체크리스트를 주고
검토. 문제 발견 → 수정 후 재실행 (검토 자체는 1회, 수정·재실행은 제한 없음).

### 6. 아카이브 문서 완성 + 검토 (1회)
2번 초안에 결과·판정·한계·재현 방법을 채워 완성 → Agent(`model: opus`)가
PROTOCOL §4-(c)로 검토 → 지적 반영.

### 7. 기록 갱신
- registry: 해당 가설을 `채택`/`기각`으로 이동 (결과 요약 + 문서 링크).
  한 시장만 통과했으면 상태는 `기각`, 결과 요약에 **"부분 지지(통과 시장 명기)"**를 남긴다
- `docs/analysis/SUMMARY.md` **맨 위**에 초보자 요약 추가 — 가설/설계/결과/함의
  4항목, 쉬운 용어, 짧게

### 8. 마무리
- git 커밋 **1개** (이번 테스트의 모든 산출물. push 금지)
- SUMMARY.md에 추가한 항목을 화면에 그대로 출력
- 캐시 경고가 있으면 함께 출력

## 흔한 실수

- 결과를 본 뒤 판정 기준을 정하기 → 반드시 3번 게이트 **전에** 고정
- 검토 sub-agent에 체크리스트 없이 "검토해줘"만 보내기 → 형식적 통과가 된다
- 유동성 필터에 `universe.dollar_vol_filter` 사용 → 최종일 기준이라 look-ahead. `btlib.liquidity.pit_dollar_vol` 사용
- registry 갱신 누락 → 다음 테스트가 중복 검사를 못 한다
