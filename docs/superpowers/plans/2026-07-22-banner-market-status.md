# 배너 시장별 데이터 기준일·상태 표시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배너가 시장별 실제 데이터 기준일과 상태(최신/장중/지연)를 보여주게 만든다.

**Architecture:** 백엔드는 `rs_asof`에 실제 마지막 봉 날짜를 기록하도록 한 줄 고친다. 프론트는 상태 판정을 순수 함수(`site/banner-status.js`)로 분리해 Node로 단위 테스트하고, `app.js`의 `renderBanner`가 그 함수를 써서 시장별 칩을 그린다. 날짜 표시는 타임존 안전한 전용 포맷터를 새로 만든다.

**Tech Stack:** Python 3.11 + pytest (파이프라인), 순수 JS(빌드 도구 없음) + Node `node:test` (프론트), GitHub Actions.

## Global Constraints

- 설계 문서: `docs/superpowers/specs/2026-07-22-banner-market-status-design.md`
- 지연 임계: **4일(24시간×4) 초과** — 달력 날짜 차이가 아니라 경과 duration (정상 최대 공백 2.40일, KR 누락 시 3.00일이므로 3일은 부족)
- 장중 판정: **평일(월–금, 시장 현지 기준) + 개장 시간대**. KR `09:00–15:30 Asia/Seoul`, US `09:30–16:00 America/New_York`
- 상태 우선순위: `stale` → `intraday` → `fresh`
- 날짜 문자열(`YYYY-MM-DD`)은 **절대 `new Date()`로 파싱하지 않는다** (뉴욕/LA에서 하루 밀림)
- 기존 `meta.warnings` 렌더링은 유지 (백엔드=시장별 실패, 프론트=사이트 정지)
- 테스트는 `daily.yml`에 넣지 않는다 (테스트 실패가 데이터 갱신을 막으면 안 됨)
- 커밋 메시지에 Claude co-author trailer 금지

---

### Task 1: `rs_asof`를 실제 마지막 봉 날짜로 기록

`rs_asof`가 지금은 `trading.latest_complete_date()`(기대 완결일)라서, 토요일 19시 KST에 `2026-07-25`(비거래일)를 반환하는 등 봉이 없는 날짜가 기록된다. 실제 데이터의 마지막 날짜로 바꾼다.

**Files:**
- Modify: `pipeline/build.py:205-214` (`build_market` 반환부)
- Modify: `pipeline/smoke.py:102`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: 없음
- Produces: `build_market()` 반환 dict의 `rs_asof`가 `frames`의 실제 최대 날짜 문자열(`"YYYY-MM-DD"`). `frames`가 비면 `meta_info["rs_asof"]`로 폴백.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_core.py` 맨 끝에 추가:

```python
# ── rs_asof = 실제 마지막 봉 날짜

def test_actual_asof_uses_last_bar_not_expected_date():
    from pipeline.build import actual_asof
    frames = {
        "A": pd.DataFrame({"date": [dt.date(2026, 7, 20), dt.date(2026, 7, 21)]}),
        "B": pd.DataFrame({"date": [dt.date(2026, 7, 17)]}),
    }
    # 기대 완결일이 토요일(07-25)이어도 실제 마지막 봉(07-21)을 써야 한다
    assert actual_asof(frames, "2026-07-25") == "2026-07-21"


def test_actual_asof_falls_back_when_no_frames():
    from pipeline.build import actual_asof
    assert actual_asof({}, "2026-07-25") == "2026-07-25"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_core.py -k actual_asof -v`
Expected: FAIL — `ImportError: cannot import name 'actual_asof'`

- [ ] **Step 3: `actual_asof` 구현**

`pipeline/build.py`의 `_group_rs` 함수 바로 위에 추가:

```python
def actual_asof(frames: dict, fallback: str) -> str:
    """frames의 실제 마지막 봉 날짜(YYYY-MM-DD). 데이터가 없으면 fallback.

    trading.latest_complete_date()는 '기대 완결일'이라 공휴일·주말엔 봉이 없는
    날짜가 나온다. 배너에 표시할 기준일은 실제 데이터 날짜여야 하므로 여기서 구한다.
    """
    last_dates = [df["date"].iloc[-1] for df in frames.values() if len(df)]
    if not last_dates:
        return fallback
    return str(max(last_dates))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_core.py -k actual_asof -v`
Expected: PASS (2 passed)

- [ ] **Step 5: `build_market`이 이 함수를 쓰도록 연결**

`pipeline/build.py`의 `build_market` 반환부에서 이 줄:

```python
        "rs_asof": meta_info["rs_asof"],
```

를 아래로 교체:

```python
        "rs_asof": actual_asof(frames, meta_info["rs_asof"]),
```

- [ ] **Step 6: `smoke.py`도 동일 문제 수정**

`pipeline/smoke.py:102`의 이 줄:

```python
        "rs_asof": str(dt.date.today()),
```

를 아래로 교체 (smoke는 US만 돌리므로 완결일 기준을 넘긴다. `build_market`이 실제 봉 날짜로 덮어쓴다):

```python
        "rs_asof": str(trading.latest_complete_date("US")),
```

그리고 `pipeline/smoke.py` 상단 import에 `trading`을 추가한다. 현재:

```python
from . import build, config, data
```

를 아래로 교체:

```python
from . import build, config, data, trading
```

- [ ] **Step 7: 전체 테스트 회귀 확인**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — `34 passed` (기존 32 + 신규 2)

- [ ] **Step 8: 커밋**

```bash
git add pipeline/build.py pipeline/smoke.py tests/test_core.py
git commit -m "fix: rs_asof를 기대 완결일이 아닌 실제 마지막 봉 날짜로 기록"
```

---

### Task 2: 배너 상태 판정 순수 함수 + Node 테스트

상태 판정 로직을 브라우저/Node 양쪽에서 쓸 수 있는 순수 함수로 만들고, 날짜/타임존 경계를 단위 테스트로 고정한다.

**Files:**
- Create: `site/banner-status.js`
- Create: `tests/banner-status.test.js` (테스트는 `site/` 밖에 둔다 — `site/`는 통째로
  GitHub Pages에 배포되므로 테스트 파일이 공개되지 않게 한다)

**Interfaces:**
- Consumes: 없음
- Produces (Task 3이 사용):
  - `fmtDateOnly(s: string) -> string` — `"2026-07-21"` → `"07-21"`. 타임존 무관.
  - `bannerStatus({market, builtAt, now}) -> {state: "stale"|"intraday"|"fresh", staleDays: number}`
  - 브라우저에서는 `window.BannerStatus.{fmtDateOnly,bannerStatus}`, Node에서는 `module.exports`로 노출.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/banner-status.test.js` 생성:

```javascript
const test = require("node:test");
const assert = require("node:assert");
const { fmtDateOnly, bannerStatus } = require("../site/banner-status.js");

// ── 날짜 포맷: 타임존이 달라도 절대 밀리지 않아야 한다
test("fmtDateOnly는 타임존과 무관하게 같은 값", () => {
  assert.strictEqual(fmtDateOnly("2026-07-21"), "07-21");
  assert.strictEqual(fmtDateOnly(""), "—");
  assert.strictEqual(fmtDateOnly(null), "—");
});

const BUILT = "2026-07-22T02:00:00Z";

// ── 장중 (KR 평일 11:00 KST = 02:00 UTC)
test("KR 평일 장중이면 intraday", () => {
  const r = bannerStatus({ market: "KR", builtAt: BUILT, now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "intraday");
});

// ── 장 마감 후 (KR 평일 16:30 KST = 07:30 UTC)
test("KR 평일 장 마감 후면 fresh", () => {
  const r = bannerStatus({ market: "KR", builtAt: BUILT, now: new Date("2026-07-22T07:30:00Z") });
  assert.strictEqual(r.state, "fresh");
});

// ── 주말: 개장 시간대여도 장중이 아니다 (요일 조건 검증)
test("토요일 10:00 KST는 intraday가 아니다", () => {
  const r = bannerStatus({ market: "KR", builtAt: "2026-07-25T00:00:00Z", now: new Date("2026-07-25T01:00:00Z") });
  assert.strictEqual(r.state, "fresh");
});

// ── US 서머타임: 평일 10:00 ET = 14:00 UTC (EDT)
test("US 평일 장중이면 intraday", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-22T13:00:00Z", now: new Date("2026-07-22T14:00:00Z") });
  assert.strictEqual(r.state, "intraday");
});

// ── 지연: 임계 4일 초과
test("built_at이 5일 지나면 stale", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-17T02:00:00Z", now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "stale");
  assert.strictEqual(r.staleDays, 5);
});

// ── 경계: KR 누락 시 최대 공백 3.00일은 stale이 아니어야 한다
test("3일 경과는 stale이 아니다 (거짓 경보 방지)", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-17T22:00:00Z", now: new Date("2026-07-20T22:00:00Z") });
  assert.notStrictEqual(r.state, "stale");
});

// ── 우선순위: 지연이 장중보다 우선
test("stale이 intraday보다 우선", () => {
  const r = bannerStatus({ market: "KR", builtAt: "2026-07-10T02:00:00Z", now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "stale");
});
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `node --test tests/banner-status.test.js`
Expected: FAIL — `Cannot find module '../site/banner-status.js'`

- [ ] **Step 3: `banner-status.js` 구현**

`site/banner-status.js` 생성:

```javascript
"use strict";

// 배너 상태 판정 — 브라우저/Node 공용 순수 함수.
// 설계: docs/superpowers/specs/2026-07-22-banner-market-status-design.md

const STALE_DAYS_LIMIT = 4; // 4 캘린더일 초과면 지연

const MARKET = {
  KR: { tz: "Asia/Seoul", open: 9 * 60, close: 15 * 60 + 30 },
  US: { tz: "America/New_York", open: 9 * 60 + 30, close: 16 * 60 },
};

// "2026-07-21" → "07-21". new Date를 쓰지 않아 타임존 영향이 없다.
function fmtDateOnly(s) {
  if (!s || typeof s !== "string") return "—";
  const parts = s.slice(0, 10).split("-");
  if (parts.length !== 3) return "—";
  return `${parts[1]}-${parts[2]}`;
}

// 특정 타임존에서의 요일(0=일)과 분 단위 시각을 구한다.
function localParts(date, timeZone) {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const parts = Object.fromEntries(fmt.formatToParts(date).map((p) => [p.type, p.value]));
  const weekdayIndex = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 }[parts.weekday];
  const hour = Number(parts.hour) % 24;
  return { weekday: weekdayIndex, minutes: hour * 60 + Number(parts.minute) };
}

function bannerStatus({ market, builtAt, now }) {
  const cfg = MARKET[market];
  const staleDays = builtAt
    ? Math.floor((now.getTime() - new Date(builtAt).getTime()) / 86400000)
    : 0;

  if (staleDays > STALE_DAYS_LIMIT) return { state: "stale", staleDays };

  if (cfg) {
    const { weekday, minutes } = localParts(now, cfg.tz);
    const isWeekday = weekday >= 1 && weekday <= 5;
    if (isWeekday && minutes >= cfg.open && minutes < cfg.close) {
      return { state: "intraday", staleDays };
    }
  }
  return { state: "fresh", staleDays };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { fmtDateOnly, bannerStatus, STALE_DAYS_LIMIT };
} else {
  window.BannerStatus = { fmtDateOnly, bannerStatus, STALE_DAYS_LIMIT };
}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `node --test tests/banner-status.test.js`
Expected: PASS — `# pass 8`, `# fail 0`

- [ ] **Step 5: 커밋**

```bash
git add site/banner-status.js tests/banner-status.test.js
git commit -m "feat: 배너 상태 판정 순수 함수 + 타임존 경계 테스트"
```

---

### Task 3: 배너에 시장별 칩 렌더링

`renderBanner`를 시장별 칩 방식으로 바꾸고, 스타일과 스크립트 로드 순서를 맞춘다.

**Files:**
- Modify: `site/index.html:352` (script 태그 추가)
- Modify: `site/app.js:54-70` (`renderBanner`)
- Modify: `site/app.css` (칩 스타일 추가)

**Interfaces:**
- Consumes: Task 2의 `window.BannerStatus.{fmtDateOnly, bannerStatus}`
- Produces: 없음 (최종 렌더)

- [ ] **Step 1: 스크립트 로드 순서 맞추기**

`site/index.html`에서 이 두 줄:

```html
  <script src="guide.js"></script>
  <script src="app.js"></script>
```

을 아래로 교체 (`banner-status.js`가 `app.js`보다 **먼저** 로드돼야 한다):

```html
  <script src="banner-status.js"></script>
  <script src="guide.js"></script>
  <script src="app.js"></script>
```

- [ ] **Step 2: `renderBanner` 교체**

`site/app.js`의 기존 `renderBanner` 함수 전체를 아래로 교체:

```javascript
const MARKET_LABEL = { US: "미국", KR: "한국" };
const STATE_LABEL = { fresh: "최신", intraday: "장중", stale: "지연" };

function renderBanner() {
  const el = document.getElementById("banner");
  const m = state.meta || {};
  const warnings = m.warnings || [];
  const rsAsof = (m.star && m.star.rs_asof) || {};
  const byCountry = m.by_country || {};
  const now = new Date();

  // 데이터가 있는 시장만 칩으로 표시
  const markets = ["US", "KR"].filter((k) => (byCountry[k] || 0) > 0);
  const chips = markets.map((k) => {
    const st = window.BannerStatus.bannerStatus({
      market: k, builtAt: m.built_at, now,
    });
    const date = window.BannerStatus.fmtDateOnly(rsAsof[k]);
    return {
      market: k, ...st, date,
      html: `<span class="mchip is-${st.state}">${MARKET_LABEL[k]} <b>${date}</b>` +
            `<span class="mchip-state">${STATE_LABEL[st.state]}</span></span>`,
    };
  });

  const stale = chips.find((c) => c.state === "stale");
  const intraday = chips.filter((c) => c.state === "intraday");
  const hasWarn = warnings.length > 0 || Boolean(stale);
  el.classList.toggle("warn", hasWarn);

  const notes = [];
  if (stale) {
    notes.push(`⚠ 데이터가 ${stale.staleDays}일째 갱신되지 않았습니다 — 배치를 확인하세요`);
  } else if (intraday.length) {
    const names = intraday.map((c) => MARKET_LABEL[c.market]).join("·");
    notes.push(`지금 ${names} 장중 — 직전 완결일 기준입니다`);
  }
  warnings.forEach((w) => notes.push(`⚠ ${w}`));

  el.innerHTML =
    `<span class="dot"></span>` +
    (chips.length
      ? `<span class="mchips">${chips.map((c) => c.html).join("")}</span>`
      : `<span>신호 기준일 <span class="asof">${fmtDate(m.built_at)}</span></span>`) +
    notes.map((n) => `<span class="warn-item">${n}</span>`).join("");
}
```

- [ ] **Step 3: 칩 스타일 추가**

`site/app.css`의 `.banner .warn-item` 줄 바로 뒤에 추가:

```css
.mchips { display: flex; gap: 8px; flex-wrap: wrap; }
.mchip {
  display: inline-flex; align-items: baseline; gap: 6px;
  font-family: var(--mono); font-size: 12px; padding: 3px 9px;
  border-radius: 7px; border: 1px solid var(--line); background: var(--panel-2);
  white-space: nowrap;
}
.mchip b { color: var(--ink); font-weight: 600; }
.mchip-state { font-size: 10.5px; opacity: 0.9; }
.mchip.is-fresh { border-color: rgba(75,184,132,0.35); color: var(--green); }
.mchip.is-intraday { border-color: rgba(242,193,78,0.35); color: var(--gold); }
.mchip.is-stale { border-color: rgba(229,104,122,0.45); color: var(--red); }
```

- [ ] **Step 4: 브라우저로 육안 확인**

```bash
cd site && python3 -m http.server 8899
```

브라우저에서 `http://localhost:8899` 접속 후 확인:
- 칩이 `미국 <날짜> 최신` / `한국 <날짜> 장중` 형태로 뜬다 (한국 장중 시간대일 때)
- 창을 좁혀도 배너가 가로로 넘치지 않고 줄바꿈된다
- 콘솔에 에러가 없다

> **날짜 기대값 주의**: 이 시점의 US 칩은 **`07-22`** 로 보인다. Task 1은 *코드*만
> 고칠 뿐 **이미 커밋된 `site/data/buy-signals.json`은 바꾸지 않기** 때문이다
> (현재 `rs_asof = {US: "2026-07-22", KR: "2026-07-21"}`). US가 실제 데이터일
> (`07-21`)로 맞춰지는 것은 Task 6 Step 5에서 US 배치를 다시 돌린 뒤다.
> **여기서 US가 07-22로 보이는 것은 정상이므로 멈추지 말 것.**

확인 후 서버 종료: `pkill -f "http.server 8899"`

- [ ] **Step 5: 커밋**

```bash
git add site/index.html site/app.js site/app.css
git commit -m "feat: 배너에 시장별 데이터 기준일·상태 칩 표시"
```

---

### Task 4: 가이드 탭 설명 동기화

배너가 바뀌었으므로 가이드 탭의 배너 설명이 사실과 어긋난다. 두 줄을 고친다.

**Files:**
- Modify: `site/index.html:133-138` (가이드 "상단 배너" 행)

**Interfaces:**
- Consumes: Task 3의 칩 UI
- Produces: 없음

- [ ] **Step 1: 가이드의 배너 설명 교체**

`site/index.html`에서 이 블록:

```html
            <div class="g-row-k">상단 배너</div>
            <div class="g-row-v">
              <b>신호 기준일</b>(데이터를 만든 날짜)이 적혀 있습니다.
              <span class="g-tag warn">붉은색 경고</span>가 뜨면 데이터 지연(신호가 묵음)이나
              약세장 같은 주의 신호가 있다는 뜻이니 참고만 하세요.
            </div>
```

을 아래로 교체:

```html
            <div class="g-row-k">상단 배너</div>
            <div class="g-row-v">
              시장별로 <b>데이터 기준일</b>과 상태가 칩으로 표시됩니다.
              <b>최신</b>=장 마감 후 최신 데이터, <b>장중</b>=지금 장이 열려 있어
              오늘치가 아직 안 들어옴(직전 완결일 기준), <b>지연</b>=데이터가 며칠째
              갱신되지 않음. <span class="g-tag warn">붉은색</span>이면 참고만 하세요.
            </div>
```

- [ ] **Step 2: 브라우저로 가이드 탭 확인**

```bash
cd site && python3 -m http.server 8899
```

`http://localhost:8899` → **데이터 읽는 법** 탭 → "목록 화면을 위에서 아래로" 섹션의
"상단 배너" 행이 새 설명으로 바뀌었는지 확인. 확인 후 `pkill -f "http.server 8899"`.

- [ ] **Step 3: 커밋**

```bash
git add site/index.html
git commit -m "docs: 가이드 탭 배너 설명을 시장별 칩 기준으로 갱신"
```

---

### Task 5: 테스트 전용 워크플로 분리

지금 CI는 테스트를 전혀 돌리지 않는다. 테스트를 추가하되 **`daily.yml`에는 넣지 않는다** — 테스트 실패가 데이터 갱신·배포를 막으면 사이트가 묵기 때문이다.

**Files:**
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: Task 1의 pytest, Task 2의 `node --test`
- Produces: 없음

- [ ] **Step 1: 워크플로 생성**

`.github/workflows/test.yml` 생성:

```yaml
name: test

# 코드 변경 시에만 테스트를 돌린다. 데이터 배치(daily.yml)와 분리해,
# 테스트 실패가 일간 데이터 갱신·배포를 막지 않도록 한다.

on:
  push:
    branches: [main]
    paths:
      - "pipeline/**"
      - "tests/**"
      - "site/**"
      - ".github/workflows/test.yml"
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      # 러너 기본 Node에 암묵적으로 의존하지 않도록 버전을 명시한다
      - uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: 의존성 설치
        run: pip install -r requirements.txt

      - name: 파이프라인 테스트 (pytest)
        run: python -m pytest -q

      - name: 프론트 테스트 (node)
        run: node --test tests/banner-status.test.js
```

- [ ] **Step 2: 로컬에서 두 테스트 명령이 통과하는지 확인**

Run:
```bash
.venv/bin/python -m pytest -q && node --test tests/banner-status.test.js
```
Expected: pytest `34 passed`, node `# fail 0`

- [ ] **Step 3: 커밋**

```bash
git add .github/workflows/test.yml
git commit -m "ci: 테스트 전용 워크플로 추가 (데일리 배치와 분리)"
```

---

### Task 6: 배포 및 라이브 검증

**Files:**
- 없음 (배포·검증만)

**Interfaces:**
- Consumes: Task 1–5
- Produces: 없음

- [ ] **Step 1: push하여 배포 트리거**

```bash
git push origin main
```

`site/**` 변경이 있으므로 `deploy-pages` 워크플로가 자동 실행된다.

- [ ] **Step 2: 배포 완료 확인**

```bash
gh run list --workflow=pages.yml --limit 1
gh run watch $(gh run list --workflow=pages.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```
Expected: `completed  success`

- [ ] **Step 3: 테스트 워크플로도 통과했는지 확인**

```bash
gh run list --workflow=test.yml --limit 1
```
Expected: `completed  success`

- [ ] **Step 4: 라이브 배너 육안 확인**

`https://itsmehank.github.io/buy-trend-site/` 접속 후:
- 시장별 칩이 뜨는지
- 날짜가 `rs_asof`와 일치하는지 (`curl -s https://itsmehank.github.io/buy-trend-site/data/buy-signals.json | python3 -c "import sys,json;print(json.load(sys.stdin)['meta']['star']['rs_asof'])"`)
- 콘솔 에러가 없는지

- [ ] **Step 5: US `rs_asof` 전환 처리**

라이브 US `rs_asof`는 구 코드가 넣은 빌드 날짜(`2026-07-22`)라, Task 1 수정 이후에도
**다음 US 배치 전까지는** 실제 데이터일(`2026-07-21`)과 다르다. 즉시 맞추려면:

```bash
gh workflow run daily.yml -f market=us
```

실행 후 `gh run watch`로 완료를 기다리고, 위 Step 4의 `rs_asof` 확인을 다시 한다.
(급하지 않으면 다음날 07:00 KST 정규 배치가 자동으로 맞춘다 — 이 경우 이 스텝은 건너뛴다.)

---

## 완료 기준

- `pytest` 34개 통과, `node --test` 8개 통과
- 라이브 배너에 시장별 칩(날짜 + 최신/장중/지연)이 표시됨
- 가이드 탭 배너 설명이 새 UI와 일치
- `test.yml`이 통과하고, `daily.yml`은 테스트와 무관하게 계속 동작
