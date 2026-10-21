#!/usr/bin/env python3
"""Upload this project to a GitHub repository through the MoClaw GitHub connector.

The connector keeps GitHub credentials server-side; this script never touches a
token. It batches files into `github.files.commit` calls so a whole tree lands
as a few reviewable commits.

Usage
-----
    # see what would be uploaded
    python scripts/upload_to_github.py --repo jackkeyburgess/quantum-nanorobot-stack --dry-run

    # upload to the default branch
    python scripts/upload_to_github.py --repo jackkeyburgess/quantum-nanorobot-stack \
        --branch main --message "feat: QNRS 0.2.0 — quantum instrument API layer"

The target repository must already exist and the connector must have repository
access (a GitHub App installation with that repo selected). Verify with:

    connector-cli provider invoke github.repository.get \
        --input '{"repo":"OWNER/NAME"}'

If that returns ``connector_account_not_found``, enable repository access for the
GitHub connector in MoClaw settings first.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "build", "dist", ".venv", "venv", ".eggs", ".idea", ".vscode",
    "install", "log",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".egg-info"}
EXCLUDE_NAMES = {".DS_Store"}


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & EXCLUDE_DIRS:
            continue
        if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
            continue
        if path.name.endswith(".egg-info"):
            continue
        yield path


def build_batches(root: Path, batch_size: int) -> List[List[Tuple[str, str]]]:
    batches: List[List[Tuple[str, str]]] = []
    batch: List[Tuple[str, str]] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        # The connector takes base64 of the file bytes; that is correct for both
        # text (UTF-8) and binary content.
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        batch.append((rel, encoded))
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def invoke(repo: str, branch: str, message: str, files: List[Tuple[str, str]],
           author_name: str, author_email: str) -> dict:
    payload = {
        "repo": repo,
        "branch": branch,
        "message": message,
        "files": [{"path": path, "content": content} for path, content in files],
    }
    if author_name and author_email:
        payload["author"] = {"name": author_name, "email": author_email}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        input_path = handle.name
    try:
        completed = subprocess.run(
            ["connector-cli", "provider", "invoke", "github.files.commit", "--input-file", input_path],
            capture_output=True, text=True, check=False,
        )
    finally:
        Path(input_path).unlink(missing_ok=True)
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = output or completed.stderr.strip() or "<no output>"
        raise SystemExit(
            f"connector-cli failed ({completed.returncode}): {detail}\n"
            "If the error is connector_account_not_found, enable repository access for the "
            "GitHub connector in MoClaw settings (install the GitHub App and select the repo)."
        )
    if not output:
        raise SystemExit("connector-cli returned no output")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise SystemExit(f"unexpected connector output: {output[:500]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload the project to GitHub via the MoClaw connector")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Add Quantum Nanorobot Runtime (QNRS) 0.2.0")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--author-name", default="")
    parser.add_argument("--author-email", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    batches = build_batches(root, args.batch_size)
    total = sum(len(batch) for batch in batches)
    print(f"project root: {root}")
    print(f"files: {total} in {len(batches)} batch(es), repo={args.repo}, branch={args.branch}")

    if args.dry_run:
        for index, batch in enumerate(batches, start=1):
            print(f"\nbatch {index} ({len(batch)} files):")
            for path, _ in batch:
                print(f"  {path}")
        return 0

    for index, batch in enumerate(batches, start=1):
        message = args.message if len(batches) == 1 else f"{args.message} [{index}/{len(batches)}]"
        result = invoke(args.repo, args.branch, message, batch, args.author_name, args.author_email)
        ok = bool(result.get("ok"))
        data = result.get("data") or {}
        commit = data.get("commit", {}).get("sha") or data.get("sha") or "?"
        status = "ok" if ok else f"FAILED: {result.get('error')}"
        print(f"batch {index}/{len(batches)}: {status} (commit {str(commit)[:12]})")
        if not ok:
            return 1
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
