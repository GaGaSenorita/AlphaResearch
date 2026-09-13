#!/usr/bin/env python3
"""Build a reproducible source-and-results ZIP without Git history or local data.

Project files come from the current working tree, including nonignored additions.
AlphaBench files come only from its pinned upstream Git objects, never its live
checkout. Run this after reviewing the working tree and completing validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile


ROOT_FILES = {
    ".env.example", ".gitattributes", ".gitignore", ".gitmodules", "README.md",
    "pyproject.toml", "environment.yml", "requirements.txt", "LICENSE",
    "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.md",
}
SOURCE_DIRS = {"configs", "src", "scripts", "tests", "docs", "integrations"}
CURATED_RUNS = {
    "factor_library",
    *(f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
      for seed in (42, 123, 456)),
}
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", "node_modules", "runtime", "logs",
    "cache", "caches", "backup", "backups", "local-only", "local_only",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".sqlite", ".sqlite3", ".db", ".pem", ".key", ".p12",
    ".pfx", ".kdbx", ".bak", ".backup", ".old", ".orig", ".zip",
}


def git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def excluded(path: PurePosixPath, *, upstream: bool = False) -> bool:
    """Exclude private/runtime material even if it was accidentally tracked."""
    parts = [part.lower() for part in path.parts]
    name = parts[-1]
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts):
        return True
    if name in {".ds_store", "credentials", "secrets", "id_rsa", "id_ed25519"}:
        return True
    if name.startswith(".env") and name != ".env.example":
        return True
    if any(name.startswith(prefix) for prefix in ("credentials.", "secrets.", "secret.")):
        return True
    # AlphaBench's pinned template uses environment placeholders and is required
    # by its loader. Its local working-tree counterpart is never read.
    if "api_keys" in name and not (upstream and path.as_posix() == "config/api_keys.yaml"):
        return True
    return (path.suffix.lower() in EXCLUDED_SUFFIXES or name.endswith("~")
            or ".sqlite-" in name or ".db-" in name)


def selected_project_path(path: PurePosixPath) -> bool:
    if excluded(path):
        return False
    if len(path.parts) == 1:
        return path.name in ROOT_FILES
    if path.parts[0] in SOURCE_DIRS:
        return True
    if path.parts[0] == "figures":
        return path.suffix.lower() in {".pdf", ".json", ".csv"}
    if path.parts[:2] != ("runs", "ldm_continuous_discovery"):
        return False
    if len(path.parts) == 3:
        return path.name in {"README.md", "results_summary.json"}
    return (path.parts[2] in CURATED_RUNS and not path.name.startswith("figure.")
            and (path.suffix.lower() in {".json", ".jsonl", ".csv", ".md"}
                 or path.name.endswith(".jsonl.gz")))


def project_files(root: Path) -> dict[str, bytes]:
    names = git_bytes(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    files = {}
    for name in sorted(set(names.decode("utf-8").split("\0")) - {""}):
        relative = PurePosixPath(name)
        if not selected_project_path(relative):
            continue
        source = root / relative
        # Never follow a file or directory link into local data or credentials.
        if any(parent.is_symlink() for parent in (source, *source.parents) if parent != root):
            continue
        if source.is_file():  # Deleted tracked files are intentionally omitted.
            files[name] = source.read_bytes()
    return files


def upstream_files(root: Path) -> tuple[dict[str, bytes], dict]:
    upstream = root / "external/AlphaBench"
    if not (upstream / ".git").exists():
        raise RuntimeError("Packaging requires the pinned AlphaBench Git checkout")
    revision = git_bytes(upstream, "rev-parse", "HEAD").decode().strip()
    integration = json.loads((root / "integrations/alphabench/manifest.json").read_text())
    if revision != integration["upstream_commit"]:
        raise RuntimeError("AlphaBench HEAD differs from the pinned integration")
    files, license_files = {}, []
    tree = git_bytes(upstream, "ls-tree", "-rz", "--full-tree", revision)
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        details, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = details.decode().split()
        name = raw_name.decode("utf-8")
        path = PurePosixPath(name)
        if kind != "blob" or mode not in {"100644", "100755"} or excluded(path, upstream=True):
            continue
        files[name] = git_bytes(upstream, "cat-file", "blob", object_id)
        if path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
            license_files.append(name)
    marker = {
        "upstream_commit": revision,
        "files": {name: hashlib.sha256(content).hexdigest()
                  for name, content in sorted(files.items())},
        "license_files": sorted(license_files),
    }
    return files, marker


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def build_submission(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Write the submission ZIP outside the repository")
    files = project_files(root)
    upstream, marker = upstream_files(root)
    files.update({f"external/AlphaBench/{name}": content for name, content in upstream.items()})
    files["external/AlphaBench/.upstream-source.json"] = json_bytes(marker)
    license_note = ("Upstream license/notice files are included unchanged.\n" if marker["license_files"]
                    else "The pinned upstream tree has no standalone LICENSE/COPYING file.\n"
                         "Its original package metadata and documentation are preserved unchanged.\n")
    files["SUBMISSION_README.txt"] = (
        "AlphaResearch submission snapshot\n\n"
        "This archive includes reviewed working-tree source and preserved pure-LDM results.\n"
        "It contains no .git history, credentials, local market data, or Stage II results.\n"
        "AlphaBench source is exported from the pinned upstream commit; its local edits\n"
        "and runtime databases are excluded. The verified patch is supplied separately.\n\n"
        "From the extracted AlphaResearch directory, using Python 3.10-3.12 and Git:\n"
        '  python -m pip install -e ".[dev,ffo]" -e ./external/AlphaBench\n'
        "  python scripts/setup_alphabench.py --apply\n"
        "  python -m pytest -q\n\n"
        "See README.md for data setup, configuration, and reproduction commands.\n"
        "SUBMISSION_MANIFEST.json records SHA256 for every other archive file.\n"
        + license_note
    ).encode("utf-8")
    manifest = {
        "format_version": 1,
        "project_revision": git_bytes(root, "rev-parse", "HEAD").decode().strip(),
        "project_source": "current tracked and nonignored working-tree files; no Git history",
        "alphabench_revision": marker["upstream_commit"],
        "files_sha256": {name: hashlib.sha256(content).hexdigest()
                         for name, content in sorted(files.items())},
    }
    files["SUBMISSION_MANIFEST.json"] = json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".submission-", suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo("AlphaResearch/" + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content, compresslevel=9)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"output": str(output), "file_count": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="ZIP destination outside the repository")
    args = parser.parse_args()
    output = args.output or args.root.resolve().parent / "AlphaResearch_submission.zip"
    print(json.dumps(build_submission(args.root, output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
