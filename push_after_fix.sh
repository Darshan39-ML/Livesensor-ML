#!/usr/bin/env bash
# simple helper to commit and push quick fixes
set -euo pipefail
msg=${1:-"chore: quick fix"}
branch=$(git rev-parse --abbrev-ref HEAD)

git add -A
if git diff --staged --quiet; then
  echo "No changes to commit"
  exit 0
fi

git commit -m "$msg"

git push origin "$branch"

echo "Pushed to $branch"
