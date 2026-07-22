# 데일리 배치 push 경합 해결 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배치 실행 중 main에 다른 push가 들어와도 그날 데이터가 버려지지 않게 한다.

**Architecture:** `daily.yml`의 커밋 단계를 테스트 가능한 셸 스크립트로 분리하고, push 거부 시 `git pull --rebase` 후 재시도하도록 만든다. 데이터 파일끼리 충돌하면 조용히 한쪽을 버리지 않고 명확히 실패시킨다.

**Tech Stack:** GitHub Actions, bash, git.

> **검토에서 걸러낸 것**: 처음엔 `checkout`을 `fetch-depth: 0`으로 바꾸려 했으나, 실제
> shallow clone(`is-shallow-repository: true`)에서 `pull --rebase`가 정상 동작하고 재시도
> push까지 성공하는 것을 확인해 제외했다. `site/data`가 12MB이고 데이터 커밋이 하루 2회
> 쌓이므로 full clone은 비용만 늘린다.

## 배경 — 실제로 일어난 일

```
09:57:50Z  예약 실행 시작 (main = 0f800d1 체크아웃)   ← cron 07:30 UTC가 2시간 27분 지연 발화
09:57:52Z  사람이 main에 push (→ a7e3ff9)
09:58:37Z  배치가 데이터 커밋 후 push → ! [rejected] main -> main (fetch first)
           build job 실패 → deploy job(needs: build) 스킵 → 그날 데이터·배포 모두 유실
```

배치 계산 자체는 성공했다. 실패한 곳은 마지막 `git push` 하나뿐인데, `daily.yml:58-63`에
`git add` → `git commit` → `git push`만 있고 **`git pull`이 없어서** non-fast-forward가 되면
그대로 죽는다. 배치는 KR 약 6분 / US 약 25분 걸리므로, 그 창에 들어온 push는 언제든 같은 결과를 만든다.

## Global Constraints

- 확인된 전제 (구현 시 이 값들을 그대로 신뢰할 것):
  - `daily.yml`은 `concurrency: group: buy-signals, cancel-in-progress: false` → **배치끼리는 직렬화**되므로 배치 vs 배치 경합은 없다. 실제 상대는 **사람/다른 워크플로의 push**다.
  - main에 push하는 워크플로는 **`daily.yml` 하나뿐**이다(`pages.yml`·`test.yml`은 push 안 함).
  - `site/data/buy-signals.json`은 **개행 없는 한 줄**이다 → 동시에 바뀌면 반드시 충돌한다(자동 병합되지 않는다).
  - 현재 `actions/checkout@v4`에 `fetch-depth` 지정이 없다 → **shallow clone(depth 1)**.
- 데이터 충돌 시 **조용히 한쪽을 채택하지 않는다.** 배치가 만든 JSON은 실행 시작 시점의
  `buy-signals.json`을 읽어 다른 시장 데이터를 병합한 결과라, 원격이 그 파일을 바꿨다면
  우리 결과는 낡은 것이다. 이 경우는 명확히 실패시킨다.
- `daily.yml`의 배치·배포 로직 자체는 건드리지 않는다. 커밋/푸시 부분만 바꾼다.
- 커밋 메시지에 Claude co-author trailer 금지.

---

### Task 1: 커밋·푸시 스크립트 분리 + 경합 시나리오 테스트

인라인 YAML 셸은 단위 테스트가 불가능하므로, 로직을 스크립트로 빼고 **실제 git 저장소로 경합을 재현해** 검증한다.

**Files:**
- Create: `scripts/commit-data.sh`
- Create: `tests/commit-data.test.sh`

**Interfaces:**
- Consumes: 없음
- Produces (Task 2가 사용): `scripts/commit-data.sh "<커밋 메시지>"` — 실행 파일.
  `site/data`를 스테이징해 변경이 있으면 커밋·푸시한다. 종료코드 0=성공(변경 없음 포함),
  1=실패(재시도 소진 또는 데이터 충돌).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/commit-data.test.sh` 생성:

```bash
#!/usr/bin/env bash
# scripts/commit-data.sh 의 경합 복구 동작 검증.
# 임시 bare 저장소를 원격으로 두고 실제 push 거부 상황을 재현한다.
set -uo pipefail

SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/scripts/commit-data.sh"
PASS=0; FAIL=0
ok()   { echo "  ok   - $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL - $1"; FAIL=$((FAIL+1)); }

setup() {                      # $1 = 작업 디렉터리
  rm -rf "$1"; mkdir -p "$1"
  # --initial-branch=main 필수: 기본값(master)이면 bare HEAD가 존재하지 않는 ref를
  # 가리켜 이후 clone이 체크아웃에 실패하고, 경합이 재현되지 않는다(거짓 통과 원인).
  git init -q --bare --initial-branch=main "$1/remote.git"
  git clone -q "$1/remote.git" "$1/work"
  cd "$1/work"
  git config user.email t@t; git config user.name t
  mkdir -p site/data; echo '{"v":0}' > site/data/buy-signals.json
  git add -A; git commit -qm init; git push -q origin main
}

# 원격에 다른 커밋을 밀어 넣는다. $1 = 파일, $2 = 내용
push_remote_change() {
  local d; d=$(mktemp -d)
  git clone -q "$TMP/remote.git" "$d/c" && cd "$d/c"
  git config user.email o@o; git config user.name o
  mkdir -p "$(dirname "$1")"; echo "$2" > "$1"
  git add -A; git commit -qm "remote change"; git push -q origin main
  cd "$TMP/work"; rm -rf "$d"
}

# ── 1) 변경 없으면 커밋 생략하고 성공
TMP=$(mktemp -d); setup "$TMP"
out=$("$SCRIPT" "no change" 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q "변경 없음" && ok "변경 없으면 커밋 생략" || bad "변경 없음 처리 (rc=$rc)"

# ── 2) 경합 없으면 그냥 커밋·푸시
TMP=$(mktemp -d); setup "$TMP"
echo '{"v":1}' > site/data/buy-signals.json
"$SCRIPT" "normal" >/dev/null 2>&1; rc=$?
remote_v=$(git --git-dir="$TMP/remote.git" show main:site/data/buy-signals.json)
[ $rc -eq 0 ] && [ "$remote_v" = '{"v":1}' ] && ok "정상 푸시" || bad "정상 푸시 (rc=$rc, remote=$remote_v)"

# ── 3) 배치 도중 '코드' push가 들어와도 복구해야 한다 (오늘 터진 시나리오)
TMP=$(mktemp -d); setup "$TMP"
echo '{"v":2}' > site/data/buy-signals.json      # 우리 데이터 변경
push_remote_change "README.md" "remote edit"      # 다른 파일이 원격에서 바뀜
"$SCRIPT" "race with code push" >/dev/null 2>&1; rc=$?
remote_v=$(git --git-dir="$TMP/remote.git" show main:site/data/buy-signals.json)
remote_r=$(git --git-dir="$TMP/remote.git" show main:README.md)
[ $rc -eq 0 ] && [ "$remote_v" = '{"v":2}' ] && [ "$remote_r" = "remote edit" ] \
  && ok "코드 push와 경합해도 양쪽 보존" || bad "코드 경합 복구 (rc=$rc, data=$remote_v, readme=$remote_r)"

# ── 4) 같은 데이터 파일이 원격에서 바뀌면 조용히 덮지 말고 실패해야 한다
TMP=$(mktemp -d); setup "$TMP"
echo '{"v":3}' > site/data/buy-signals.json
push_remote_change "site/data/buy-signals.json" '{"v":99}'
"$SCRIPT" "data conflict" >/dev/null 2>&1; rc=$?
remote_v=$(git --git-dir="$TMP/remote.git" show main:site/data/buy-signals.json)
[ $rc -ne 0 ] && [ "$remote_v" = '{"v":99}' ] \
  && ok "데이터 충돌은 실패시키고 원격 보존" || bad "데이터 충돌 처리 (rc=$rc, remote=$remote_v)"

cd /; echo; echo "pass=$PASS fail=$FAIL"
[ $FAIL -eq 0 ]
```

실행 권한 부여:

```bash
chmod +x tests/commit-data.test.sh
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `bash tests/commit-data.test.sh`
Expected: FAIL — 스크립트가 없어 4개 케이스 모두 실패, 마지막 줄 `pass=0 fail=4`

- [ ] **Step 3: `scripts/commit-data.sh` 구현**

```bash
#!/usr/bin/env bash
# site/data 변경을 커밋·푸시한다.
#
# 배치는 KR 약 6분 / US 약 25분 걸리므로 그 사이에 main이 움직일 수 있다.
# push가 거부되면 pull --rebase 후 재시도한다. 단, 데이터 파일끼리 충돌하면
# 우리 JSON은 실행 시작 시점 데이터를 병합한 낡은 결과이므로 조용히 덮지 않고 실패시킨다.
set -uo pipefail

MSG="${1:?커밋 메시지가 필요합니다}"
ATTEMPTS=3

git add site/data
if git diff --cached --quiet; then
  echo "변경 없음 — 커밋 생략"
  exit 0
fi
git commit -m "$MSG"

for attempt in $(seq 1 "$ATTEMPTS"); do
  if git push; then
    echo "푸시 성공 (시도 $attempt)"
    exit 0
  fi
  echo "푸시 거부 — 원격 변경을 반영해 재시도합니다 ($attempt/$ATTEMPTS)"
  if ! git pull --rebase --no-edit; then
    git rebase --abort 2>/dev/null || true
    echo "::error::site/data 가 원격에서도 변경돼 rebase 충돌. 배치 결과가 낡았을 수 있어 중단합니다." >&2
    exit 1
  fi
  sleep 3
done

echo "::error::${ATTEMPTS}회 재시도 후에도 푸시 실패" >&2
exit 1
```

실행 권한 부여 (git이 실행 비트를 보존하므로 커밋 전에 반드시 실행):

```bash
chmod +x scripts/commit-data.sh
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `bash tests/commit-data.test.sh`
Expected: PASS — 마지막 줄 `pass=4 fail=0`, 종료코드 0

- [ ] **Step 5: 실행 비트가 실제로 커밋되는지 확인**

Run: `git add scripts/commit-data.sh tests/commit-data.test.sh && git diff --cached --stat && git ls-files -s scripts/commit-data.sh`
Expected: 모드가 `100755`로 표시됨 (`100644`면 `chmod +x` 후 다시 `git add`)

- [ ] **Step 6: 커밋**

```bash
git add scripts/commit-data.sh tests/commit-data.test.sh
git commit -m "feat: 데이터 커밋·푸시를 경합 복구 가능한 스크립트로 분리"
```

---

### Task 2: `daily.yml`이 스크립트를 쓰도록 교체

**Files:**
- Modify: `.github/workflows/daily.yml` (커밋 단계만)

**Interfaces:**
- Consumes: Task 1의 `scripts/commit-data.sh`
- Produces: 없음

- [ ] **Step 1: 커밋 단계를 스크립트 호출로 교체**

`.github/workflows/daily.yml`의 이 블록:

```yaml
      - name: 산출 JSON 커밋
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add site/data
          if git diff --cached --quiet; then
            echo "변경 없음 — 커밋 생략"
          else
            git commit -m "data: ${{ steps.market.outputs.market }} 갱신 ($(date -u +%FT%TZ))"
            git push
          fi
```

을 아래로 교체:

```yaml
      - name: 산출 JSON 커밋
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          ./scripts/commit-data.sh "data: ${{ steps.market.outputs.market }} 갱신 ($(date -u +%FT%TZ))"
```

- [ ] **Step 2: YAML 유효성 확인**

Run: `/Users/hank.es/git/personal/buy-copy/.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml')); print('yaml ok')"`
Expected: `yaml ok`

- [ ] **Step 3: 커밋 단계 외에는 손대지 않았는지 확인**

Run: `git diff --stat .github/workflows/daily.yml`
Expected: `.github/workflows/daily.yml` 한 파일만, 커밋 단계만 축약됨(checkout은 건드리지 않음)

- [ ] **Step 4: 커밋**

```bash
git add .github/workflows/daily.yml
git commit -m "fix: 배치 push 경합 시 rebase 후 재시도"
```

---

### Task 3: 셸 테스트를 CI에 추가

`test.yml`이 지금은 pytest와 node만 돌린다. 새 셸 테스트도 돌려야 회귀를 잡는다.

**Files:**
- Modify: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: Task 1의 `tests/commit-data.test.sh`
- Produces: 없음

- [ ] **Step 1: 트리거 경로에 `scripts/**` 추가**

`.github/workflows/test.yml`의 `paths:` 목록에서 이 줄 뒤에:

```yaml
      - "tests/**"
```

아래 줄을 추가:

```yaml
      - "scripts/**"
```

- [ ] **Step 2: 테스트 스텝 추가**

`.github/workflows/test.yml`의 마지막 스텝:

```yaml
      - name: 프론트 테스트 (node)
        run: node --test tests/banner-status.test.js
```

뒤에 아래를 추가:

```yaml
      - name: 배치 커밋 스크립트 테스트 (bash)
        run: bash tests/commit-data.test.sh
```

- [ ] **Step 3: 전체 테스트가 로컬에서 통과하는지 확인**

Run:
```bash
/Users/hank.es/git/personal/buy-copy/.venv/bin/python -m pytest -q \
  && node --test tests/banner-status.test.js \
  && bash tests/commit-data.test.sh
```
Expected: pytest `34 passed`, node `fail 0`, bash `pass=4 fail=0`

- [ ] **Step 4: 커밋**

```bash
git add .github/workflows/test.yml
git commit -m "ci: 배치 커밋 스크립트 셸 테스트를 CI에 추가"
```

---

### Task 4: 배포 및 실제 경합 재현 검증

**Files:** 없음 (검증만)

- [ ] **Step 1: push**

```bash
git push origin main
```

- [ ] **Step 2: test 워크플로 통과 확인**

```bash
gh run watch $(gh run list --workflow=test.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```
Expected: `completed  success`, 그리고 "배치 커밋 스크립트 테스트 (bash)" 스텝이 success

- [ ] **Step 3: 실제 배치가 새 커밋 경로로 도는지 확인**

```bash
gh workflow run daily.yml -f market=kr
sleep 8
gh run watch $(gh run list --workflow=daily.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```
Expected: `completed success`. 로그의 "산출 JSON 커밋" 스텝에 `푸시 성공 (시도 1)` 또는
`변경 없음 — 커밋 생략`이 보인다.

- [ ] **Step 4: 실패했던 시나리오가 이제 복구되는지 실증**

배치가 도는 도중에 main에 무해한 push를 넣어 오늘 터진 경합을 재현한다.

```bash
# 배치 시작
gh workflow run daily.yml -f market=kr
sleep 45                                   # 배치가 가격 다운로드 중일 때
# 경합 유발: 임시 파일을 push (검증 후 되돌린다)
echo "race-check $(date -u +%FT%TZ)" > docs/.race-check
git add docs/.race-check && git commit -m "test: push 경합 재현용 임시 파일" && git push
# 배치 결과 확인
gh run watch $(gh run list --workflow=daily.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```
Expected: 배치가 **success**로 끝나고, "산출 JSON 커밋" 로그에
`푸시 거부 — 원격 변경을 반영해 재시도합니다 (1/3)` 다음 `푸시 성공 (시도 2)`가 보인다.
(수정 전이라면 여기서 실패했을 것)

- [ ] **Step 5: 재현용 임시 파일 정리**

```bash
git pull --ff-only
git rm -q docs/.race-check && git commit -m "chore: 경합 재현용 임시 파일 제거" && git push
```

- [ ] **Step 6: 데이터 무결성 확인**

```bash
git pull --ff-only
curl -s "https://itsmehank.github.io/buy-trend-site/data/buy-signals.json" \
  | /Users/hank.es/git/personal/buy-copy/.venv/bin/python -c "import sys,json; d=json.load(sys.stdin); m=d['meta']; print(d['count'], m['by_country'], m['star']['rs_asof'])"
```
Expected: 행 수·`by_country`·`rs_asof`가 배치 로그와 일치한다.

---

## 완료 기준

- `bash tests/commit-data.test.sh` 4개 케이스 통과, CI에서도 실행됨
- 배치 도중 main에 push가 들어와도 배치가 성공하고 양쪽 변경이 모두 보존됨
- 데이터 파일이 원격에서도 바뀐 경우엔 조용히 덮지 않고 실패함
