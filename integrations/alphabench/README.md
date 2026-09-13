# AlphaBench integration

Upstream is the submodule at `external/AlphaBench`, pinned to
`31bb94bbb7744177c51c9d011e07c31e3092b93e`. This directory owns only the local
integration delta, not another copy of AlphaBench.

`verified-evaluation.patch` is byte-for-byte equivalent to the former two
`overlays/alphabench/ffo/` files when applied to that revision. `manifest.json`
records the patch checksum and the before/after SHA-256 of each target.

The existing changes ensure that:

- execution failures and empty/incomplete daily records are not reported as
  measured zero RankIC;
- cache-disabled evaluation is isolated from persistent cached measurements;
- combined evaluation checks all requested factors and its daily records.

Metric definitions, returns, time splits and ranking logic are unchanged.
The client independently verifies responses in `alpha_research/evaluator.py`.

From the repository root:

```bash
python scripts/setup_alphabench.py          # inspect only
python scripts/setup_alphabench.py --apply  # apply explicitly before starting FFO
```

Setup accepts either the exact upstream files or the exact patched files. It is
idempotent and refuses different revisions, modified files and partially applied
patches. It does not read credentials, start a server, or run an evaluation.
The tests apply the patch to isolated temporary directories and check both final
hashes. Never apply it underneath a running FFO process; stop and restart that
service as part of a deliberate deployment.

After applying, `git -C external/AlphaBench status --short` will correctly show
the two modified vendor files. Keep the submodule revision pinned; do not commit
the patched tree as an unrelated upstream revision.

## Source-only submission ZIP

The packaged dependency contains pristine files from the pinned upstream revision,
plus `.upstream-source.json` recording file hashes. No Git metadata, live caches or
local credentials are included. The runtime verifies this manifest when the
directory has no own `.git`; the same checksummed integrity patch can then be
applied using the setup command above. Only the documented patched hashes are
accepted in place of the corresponding pristine files.

The pinned upstream `pyproject.toml` declares MIT licensing but the upstream tree
does not contain a standalone LICENSE file. The export preserves its original
metadata and attribution without inventing additional licence text.
