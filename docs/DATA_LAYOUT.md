# Data organisation

## Versioned evidence

`runs/ldm_continuous_discovery/` contains the three preserved pure LDM R100 result
snapshots, their provenance and the derived cross-seed factor library. Scientific
measurements are retained byte-for-byte. Canonical figure PDFs and their JSON/CSV
inputs live in `figures/`; per-run copies and alternate formats are unnecessary.

## Local-only results

Stage II output defaults to the sibling directory:

```text
AlphaResearch_local_results/
  stage2/
    ldm_split_robust_reward/       Annual Method 1 outputs
    ldm_rankic_worst_qehvi/        Annual Method 2 outputs, when available
    legacy_quarterly/             Original quarterly searches, kept unchanged
    migration_manifest.json       Original/destination paths and file SHA256
  archived_figure_duplicates/     Removed duplicate exports, safely retained
```

The local manifest records the actual partition and completion state. Partial
runs remain partial. Old quarterly results must not be presented as results of the
current annual methods, even if daily measurements can be aggregated by year.
The original raw files are preserved; changing the implementation does not rerun
the historical experiment.

Stage II methods, configurations and tests are versioned; their run data are not.
Generated runtime files, environments and market data are likewise excluded.
The submission packager independently enforces these exclusions even if an
excluded path has accidentally been staged or tracked.

## Reproducible delivery

`python scripts/package_submission.py` creates `../AlphaResearch_submission.zip`
from the current working source and the selected LDM evidence. It includes an
internal SHA256 manifest and pristine AlphaBench source from the pinned revision,
not the live dependency's caches, database files or credentials. No Git history
is included. Existing Git history is retained normally in the development repo.
