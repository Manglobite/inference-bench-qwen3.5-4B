#!/usr/bin/env bash
# Portable publication barrier without git: recursively scan the case tree for
# private markers and enforce a strict file allowlist. The script excludes
# itself from marker scanning, since its own pattern list contains the very
# literals it looks for. Run before publishing.
#
# Fail-closed: an unexpected file, a scan error or any match exits 2; exit 0
# only when clean.

set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

SELF="./scripts/check-public.sh"
PATTERN='package api|package service|internal/api/|internal/service/|internal/model/|internal/store/|webUI/src/|/home/|/root/|/mnt/|/var/backups/|manglobite|sk-[A-Za-z0-9]{13,}|llm-limits-proxy|localhost|127\.0\.0\.1|ses_[a-z0-9]|[0-9]{1,3}(\.[0-9]{1,3}){3}|@lab\b'
ALLOWLIST='^\./(README\.md|README\.en\.md|LICENSE-NOTICE\.md|\.gitignore|scripts/check-public\.sh|data/[A-Za-z0-9._-]+\.svg)$'
# The companion case repository is public and was published by the owner, so
# this exact URL is allowed on purpose. Only this literal URL is stripped
# before re-checking; bare `manglobite` occurrences outside it still fail.
ALLOWED_URL='https://github.com/Manglobite/inference-cmp70-tuning'

mapfile -d '' -t scan < <(find . -type f ! -path './.git/*' ! -path "$SELF" -print0)

if [[ ${#scan[@]} -gt 0 ]]; then
  set +e
  hits="$(grep -nHIiE --binary-files=without-match -e "$PATTERN" -- "${scan[@]}" 2>/dev/null)"
  status=$?
  set -e

  if [[ "$status" -gt 1 ]]; then
    echo "FAIL: scan error (grep status $status)" >&2
    exit 2
  fi

  if [[ -n "$hits" ]]; then
    residual="$(printf '%s\n' "$hits" | sed "s|$ALLOWED_URL||g" | grep -iE -e "$PATTERN" || true)"
    if [[ -n "$residual" ]]; then
      printf '%s\n' "$residual" >&2
      echo "FAIL: private markers found" >&2
      exit 2
    fi
  fi
fi

mapfile -d '' -t files < <(find . -type f ! -path './.git/*' -print0)

unexpected=0
for f in "${files[@]}"; do
  if [[ ! "$f" =~ $ALLOWLIST ]]; then
    echo "FAIL: unexpected file: $f" >&2
    unexpected=1
  fi
done

if [[ "$unexpected" -ne 0 ]]; then
  exit 2
fi

echo "OK: no private markers found"
