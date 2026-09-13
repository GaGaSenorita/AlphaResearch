"""Exercise the actual ZIP boundary with independent, temporary Git checkouts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_submission.py"
SPEC = importlib.util.spec_from_file_location("package_submission", SCRIPT)
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()


def write(root, name, content="content\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def initialise(root):
    root.mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Packaging test")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    initialise(root)
    write(root, "README.md", "Initial README\n")
    write(root, "pyproject.toml", "[project]\nname='fixture'\n")
    write(root, ".gitignore", "docs/ignored.md\n")
    write(root, ".env.example", "API_KEY=\n")
    write(root, ".env", "LOCAL_SECRET=private\n")
    write(root, "src/fixture/module.py", "version = 'committed'\n")
    write(root, "src/fixture/deleted.py")
    for name in ("src/fixture/__pycache__/cached.pyc", "docs/backup/old.md",
                 "configs/credentials.json", "configs/private.key", "runtime/results.json",
                 "runs/ldm_split_robust_reward/result.json", "runs/ldm_rankic_worst_qehvi/result.json",
                 "runs/ldm_continuous_discovery/unreviewed_seed/result.json"):
        write(root, name, "must never be packaged\n")
    for ext in ("pdf", "json", "csv", "png", "svg"):
        write(root, f"figures/trajectory.{ext}")
    run = "runs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed42"
    write(root, f"{run}/resume_state.json", '{"round_id":100}\n')
    write(root, f"{run}/figure.pdf")
    write(root, f"{run}/figure.json")
    write(root, "runs/ldm_continuous_discovery/factor_library/admitted_factors.csv")

    upstream = root / "external/AlphaBench"
    initialise(upstream)
    write(upstream, "LICENSE", "Original upstream license notice\n")
    write(upstream, "pyproject.toml", "[project]\nname='upstream-fixture'\n")
    write(upstream, "agent/source.py", "state = 'pristine upstream'\n")
    write(upstream, "config/api_keys.yaml", 'api_key: "${API_KEY}"\n')
    write(upstream, "factor_cache.sqlite", "tracked cache\n")
    write(upstream, "ffo/.DS_Store", "tracked Finder metadata\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "upstream fixture")
    revision = git(upstream, "rev-parse", "HEAD")
    write(root, "integrations/alphabench/manifest.json", json.dumps({"upstream_commit": revision}))
    git(root, "add", ".")
    git(root, "commit", "-qm", "project fixture")

    write(root, "README.md", "Reviewed working-tree README\n")
    write(root, "src/fixture/module.py", "version = 'reviewed working tree'\n")
    write(root, "scripts/new_script.py", "print('new nonignored source')\n")
    write(root, "docs/ignored.md", "untracked ignored notes\n")
    (root / "src/fixture/deleted.py").unlink()
    (root / "src/fixture/linked_secret.py").symlink_to(root / ".env")
    write(upstream, "agent/source.py", "state = 'local uncommitted edits'\n")
    write(upstream, "config/api_keys.yaml", 'api_key: "local private credential"\n')
    write(upstream, "config/credentials.json", "local private credential\n")
    write(upstream, "ffo/factor_cache.sqlite-wal", "live cache\n")
    return root


def test_archive_contains_working_tree_source_and_only_curated_data(project, tmp_path):
    result = PACKAGER.build_submission(project, tmp_path / "submission.zip")
    with zipfile.ZipFile(result["output"]) as archive:
        files = {name.removeprefix("AlphaResearch/"): archive.read(name) for name in archive.namelist()}
    assert files["README.md"] == b"Reviewed working-tree README\n"
    assert files["src/fixture/module.py"] == b"version = 'reviewed working tree'\n"
    assert "scripts/new_script.py" in files
    assert ".env.example" in files
    assert {"figures/trajectory.pdf", "figures/trajectory.json", "figures/trajectory.csv"} <= files.keys()
    assert "runs/ldm_continuous_discovery/factor_library/admitted_factors.csv" in files
    run = "runs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed42"
    assert f"{run}/resume_state.json" in files
    assert not any(name.startswith(("runtime/", "runs/ldm_split_robust_reward/",
                                    "runs/ldm_rankic_worst_qehvi/")) for name in files)
    excluded = {".env", "src/fixture/deleted.py", "src/fixture/linked_secret.py",
                "src/fixture/__pycache__/cached.pyc", "docs/backup/old.md", "docs/ignored.md",
                "configs/credentials.json", "configs/private.key", "figures/trajectory.png",
                "figures/trajectory.svg", f"{run}/figure.pdf", f"{run}/figure.json",
                "runs/ldm_continuous_discovery/unreviewed_seed/result.json"}
    assert excluded.isdisjoint(files)
    assert not any(".git" in Path(name).parts for name in files)
    assert not any(b"local private credential" in content for content in files.values())


def test_upstream_comes_from_git_objects_with_license_and_verified_marker(project, tmp_path):
    result = PACKAGER.build_submission(project, tmp_path / "submission.zip")
    with zipfile.ZipFile(result["output"]) as archive:
        prefix = "AlphaResearch/external/AlphaBench/"
        assert archive.read(prefix + "agent/source.py") == b"state = 'pristine upstream'\n"
        assert archive.read(prefix + "LICENSE") == b"Original upstream license notice\n"
        assert archive.read(prefix + "config/api_keys.yaml") == b'api_key: "${API_KEY}"\n'
        names = archive.namelist()
        assert not any(name.endswith(("factor_cache.sqlite", "sqlite-wal", ".DS_Store", "credentials.json"))
                       for name in names)
        marker = json.loads(archive.read(prefix + ".upstream-source.json"))
        assert marker["upstream_commit"] == git(project / "external/AlphaBench", "rev-parse", "HEAD")
        assert marker["license_files"] == ["LICENSE"]
        exported = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
        assert set(marker["files"]) == exported - {".upstream-source.json"}
        for name, digest in marker["files"].items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest


def test_zip_is_reproducible_and_manifest_covers_every_payload(project, tmp_path):
    first = PACKAGER.build_submission(project, tmp_path / "first.zip")
    os.utime(project / "README.md", (1700000000, 1700000000))
    second = PACKAGER.build_submission(project, tmp_path / "second.zip")
    assert Path(first["output"]).read_bytes() == Path(second["output"]).read_bytes()
    with zipfile.ZipFile(first["output"]) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert archive.testzip() is None
        manifest = json.loads(archive.read("AlphaResearch/SUBMISSION_MANIFEST.json"))
        assert set(manifest["files_sha256"]) == {
            name.removeprefix("AlphaResearch/") for name in archive.namelist()
        } - {"SUBMISSION_MANIFEST.json"}
        for name, digest in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read("AlphaResearch/" + name)).hexdigest() == digest
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_missing_upstream_license_is_reported_without_fabrication(project, tmp_path):
    upstream = project / "external/AlphaBench"
    git(upstream, "rm", "LICENSE")
    git(upstream, "commit", "-qm", "remove notice from fixture")
    write(project, "integrations/alphabench/manifest.json",
          json.dumps({"upstream_commit": git(upstream, "rev-parse", "HEAD")}))
    result = PACKAGER.build_submission(project, tmp_path / "submission.zip")
    with zipfile.ZipFile(result["output"]) as archive:
        assert "AlphaResearch/external/AlphaBench/LICENSE" not in archive.namelist()
        assert b"no standalone LICENSE" in archive.read("AlphaResearch/SUBMISSION_README.txt")


def test_wrong_upstream_revision_or_in_repo_destination_is_rejected(project, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        PACKAGER.build_submission(project, project / "submission.zip")
    upstream = project / "external/AlphaBench"
    git(upstream, "commit", "--allow-empty", "-qm", "unexpected upstream revision")
    with pytest.raises(RuntimeError, match="pinned"):
        PACKAGER.build_submission(project, tmp_path / "submission.zip")
    assert not (tmp_path / "submission.zip").exists()
