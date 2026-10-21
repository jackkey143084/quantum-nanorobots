#!/usr/bin/env bash
# Fallback upload path: run this on YOUR machine (where you are authenticated to
# GitHub) if the MoClaw connector does not have repository access.
#
# Usage:
#   1. Download qnrs-0.2.0.bundle (published alongside the zip artifact).
#   2. bash push_from_local.sh /path/to/qnrs-0.2.0.bundle
#
# The bundle already contains the full commit history, so this is a plain
# `git clone` + `git push` — no rebuild, no re-upload of 64 files by hand.
set -euo pipefail

BUNDLE="${1:?usage: push_from_local.sh /path/to/qnrs-0.2.0.bundle}"
REPO_URL="${REPO_URL:-https://github.com/jackkeyburgess/Nanorobots_quantum.git}"
WORKDIR="$(mktemp -d)"

echo ">> cloning bundle into $WORKDIR/qnrs"
git clone --quiet "$BUNDLE" "$WORKDIR/qnrs"
cd "$WORKDIR/qnrs"

echo ">> target: $REPO_URL (branch main)"
git remote set-url origin "$REPO_URL" 2>/dev/null || git remote add origin "$REPO_URL"

echo ">> pushing"
# The remote 'main' contains only GitHub's auto-generated README, so its history
# is unrelated to ours. --force is required and replaces that single commit.
git push -u origin main --force

echo ">> done: $REPO_URL"
