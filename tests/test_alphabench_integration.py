"""Exercise the real integration patch in isolated temporary checkouts only."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from alpha_research import alphabench_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


def _upstream_bytes(name):
    """Read pristine fixtures from a checkout or the distributable source ZIP."""
    source = ROOT / "external/AlphaBench"
    if (source / ".git").exists():
        return subprocess.check_output([
            "git", "-C", str(source), "show", f"{runtime.PINNED_COMMIT}:{name}",
        ])
    content = (source / name).read_bytes()
    manifest = json.loads((runtime.INTEGRATION_DIR / "manifest.json").read_text())
    hashes = manifest["files"].get(name)
    if hashes and hashlib.sha256(content).hexdigest() == hashes["patched_sha256"]:
        # A recipient may have applied the integration already. Recover the
        # pristine fixture in a temporary directory, never undo their setup.
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary)
            for target_name in manifest["files"]:
                target = copy / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / target_name).read_bytes())
            subprocess.run([
                "git", "-C", str(copy), "apply", "--no-index", "--reverse",
                str(runtime.INTEGRATION_DIR / "verified-evaluation.patch"),
            ], check=True)
            content = (copy / name).read_bytes()
    if hashes:
        assert hashlib.sha256(content).hexdigest() == hashes["upstream_sha256"]
    return content


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    checkout = tmp_path / "AlphaBench"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    manifest = json.loads((runtime.INTEGRATION_DIR / "manifest.json").read_text())
    for name in manifest["files"]:
        content = _upstream_bytes(name)
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


@pytest.fixture
def source_export(tmp_path):
    checkout = tmp_path / "source-export"
    manifest = json.loads((runtime.INTEGRATION_DIR / "manifest.json").read_text())
    names = {
        *manifest["files"], "searcher/algo/cot.py", "ffo/client/factor_eval_client.py",
        *(f"factors/lib/alpha158/{name}" for name in (
            "__init__.py", "qlib_compile_product.json", "kbar.json", "price.json", "rolling.json",
        )),
    }
    files = {}
    for name in sorted(names):
        content = _upstream_bytes(name)
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    (checkout / runtime.SOURCE_EXPORT_MARKER).write_text(json.dumps({
        "upstream_commit": runtime.PINNED_COMMIT, "files": files,
    }))
    return checkout


def test_source_export_loads_the_actual_alpha158_library_without_git(source_export):
    assert runtime.checkout_revision(source_export) == runtime.PINNED_COMMIT
    assert runtime.resolve_alphabench_root(source_export) == source_export
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from pathlib import Path; "
        "from alpha_research.alphabench_runtime import load_alpha158_seeds; "
        "seeds = load_alpha158_seeds(sys.argv[1]); "
        "import factors.lib.alpha158 as source; "
        "assert Path(source.__file__).resolve().is_relative_to(Path(sys.argv[1])); "
        "assert len(seeds) == 42; print(len(seeds))",
        str(source_export),
    ], cwd=source_export.parent, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_git_free_export_patch_is_idempotent_even_inside_another_repository(source_export):
    subprocess.run(["git", "init", "-q", str(source_export.parent)], check=True)
    marker_before = (source_export / runtime.SOURCE_EXPORT_MARKER).read_bytes()
    assert runtime.configure_verified_evaluator(source_export) == "upstream"
    assert runtime.configure_verified_evaluator(source_export, apply=True) == "patched"
    assert runtime.configure_verified_evaluator(source_export, apply=True) == "patched"
    assert runtime.checkout_revision(source_export) == runtime.PINNED_COMMIT
    assert (source_export / runtime.SOURCE_EXPORT_MARKER).read_bytes() == marker_before


def test_git_free_export_rejects_mixed_integration_states(source_export):
    runtime.configure_verified_evaluator(source_export, apply=True)
    name = "ffo/routes/factors.py"
    (source_export / name).write_bytes(_upstream_bytes(name))
    with pytest.raises(RuntimeError, match="Partially patched"):
        runtime.configure_verified_evaluator(source_export, apply=True)


@pytest.mark.parametrize("already_patched", [False, True])
def test_pristine_test_fixtures_work_in_both_export_setup_states(source_export, monkeypatch, already_patched):
    project = source_export.parent / "exported-project"
    checkout = project / "external" / "AlphaBench"
    checkout.parent.mkdir(parents=True)
    source_export.rename(checkout)
    monkeypatch.setitem(globals(), "ROOT", project)
    if already_patched:
        runtime.configure_verified_evaluator(checkout, apply=True)
    manifest = json.loads((runtime.INTEGRATION_DIR / "manifest.json").read_text())
    before = {name: (checkout / name).read_bytes() for name in manifest["files"]}
    for name, hashes in manifest["files"].items():
        assert hashlib.sha256(_upstream_bytes(name)).hexdigest() == hashes["upstream_sha256"]
    assert all((checkout / name).read_bytes() == content for name, content in before.items())


@pytest.mark.parametrize("damage", ["commit", "contents", "checksum", "missing", "traversal", "symlink"])
def test_source_export_rejects_bad_identity_or_tampering(source_export, damage):
    marker = source_export / runtime.SOURCE_EXPORT_MARKER
    metadata = json.loads(marker.read_text())
    name = "factors/lib/alpha158/kbar.json"
    if damage == "commit":
        metadata["upstream_commit"] = "0" * 40
    elif damage == "contents":
        (source_export / name).write_bytes(b"changed source")
    elif damage == "checksum":
        metadata["files"][name] = "0" * 64
    elif damage == "missing":
        del metadata["files"]["ffo/routes/factors.py"]
    elif damage == "traversal":
        metadata["files"]["../outside.py"] = "0" * 64
    elif damage == "symlink":
        target = source_export / name
        outside = source_export.parent / "outside.json"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
    marker.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="source.export"):
        runtime.checkout_revision(source_export)


def test_missing_export_marker_never_uses_parent_git_revision(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    checkout = tmp_path / "no-metadata"
    checkout.mkdir()
    with pytest.raises(RuntimeError, match="no own .git metadata"):
        runtime.checkout_revision(checkout)
