#!/bin/sh
# Verify the frontend from only files represented in a Git archive.
# Usage: scripts/verify-frontend-clean-build.sh [archive]
# Without an argument, the script archives the current working tree (including
# uncommitted changes) through a temporary Git index.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
archive=${1:-}

cleanup() {
  if [ -n "${workdir:-}" ]; then
    rm -rf "$workdir"
  fi
}
trap cleanup EXIT HUP INT TERM

workdir=$(mktemp -d "${TMPDIR:-/tmp}/pneumonia-frontend-clean.XXXXXX")
archive_dir="$workdir/repository"
mkdir -p "$archive_dir"

if [ -n "$archive" ]; then
  # Pass an archive produced by `git archive` from the repository root.
  tar -xf "$archive" -C "$archive_dir"
else
  # Build a tree from the current checkout without changing its real index.
  # This makes the script useful before a commit while still using Git's archive
  # machinery and excluding ignored build/dependency directories.
  index_file="$workdir/index"
  GIT_INDEX_FILE="$index_file" git -C "$repo_root" read-tree HEAD
  (cd "$repo_root" && GIT_INDEX_FILE="$index_file" git add -A -- .)
  tree=$(GIT_INDEX_FILE="$index_file" git -C "$repo_root" write-tree)
  git -C "$repo_root" archive --format=tar "$tree" | tar -xf - -C "$archive_dir"
fi

cd "$archive_dir/frontend"
npm ci
npm run typecheck
npm run build

test -f "$archive_dir/frontend/app/lib/api.ts"
test -f "$archive_dir/frontend/app/lib/types.ts"
test -d "$archive_dir/frontend/out"
