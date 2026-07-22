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

# ── 3) 배치 도중 '코드' push가 들어와도 복구해야 한다 (실제로 터진 시나리오)
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
