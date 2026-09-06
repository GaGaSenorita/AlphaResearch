"""Exercise the real integration patch in isolated temporary checkouts only."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from alpha_research import alphabench_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    checkout = tmp_path / "AlphaBench"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    manifest = json.loads((runtime.INTEGRATION_DIR / "manifest.json").read_text())
    for name in manifest["files"]:
        content = subprocess.check_output([
            "git", "-C", str(ROOT / "external/AlphaBench"),
            "show", f"{runtime.PINNED_COMMIT}:{name}",
        ])
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(runtime, "checkout_revision", lambda _: runtime.PINNED_COMMIT)
    return checkout, manifest


def test_patch_is_exact_idempotent_and_check_is_read_only(upstream):
    root, manifest = upstream
    assert runtime.configure_verified_evaluator(root) == "upstream"
    for name, hashes in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == hashes["upstream_sha256"]
    assert runtime.configure_verified_evaluator(root, apply=True) == "patched"
    assert runtime.configure_verified_evaluator(root, apply=True) == "patched"
    for name, hashes in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == hashes["patched_sha256"]


def test_local_modification_does_not_get_overwritten(upstream):
    root, _ = upstream
    edited = root / "ffo/utils/utils.py"
    edited.write_bytes(edited.read_bytes() + b"\n# local change\n")
    before = {p: p.read_bytes() for p in root.rglob("*.py")}
    with pytest.raises(RuntimeError, match="Local modifications"):
        runtime.configure_verified_evaluator(root, apply=True)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_wrong_revision_is_rejected(upstream, monkeypatch):
    root, _ = upstream
    monkeypatch.setattr(runtime, "checkout_revision", lambda _: "0" * 40)
    with pytest.raises(RuntimeError, match="revision differs"):
        runtime.configure_verified_evaluator(root, apply=True)


def test_partial_patch_is_rejected(upstream):
    root, _ = upstream
    path = root / "ffo/routes/factors.py"
    original = path.read_bytes()
    runtime.configure_verified_evaluator(root, apply=True)
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="Partially patched"):
        runtime.configure_verified_evaluator(root, apply=True)


@pytest.mark.parametrize("separate_git_dir", [False, True])
def test_directory_and_git_file_revisions_are_resolved(tmp_path, separate_git_dir):
    root = tmp_path / "checkout"
    command = ["git", "init", "-q"]
    if separate_git_dir:
        command += ["--separate-git-dir", str(tmp_path / "metadata")]
    subprocess.run([*command, str(root)], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.name=Test",
        "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-qm", "test revision", "--allow-empty",
    ], check=True)
    expected = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert (root / ".git").is_file() == separate_git_dir
    assert runtime.checkout_revision(root) == expected
