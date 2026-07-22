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
