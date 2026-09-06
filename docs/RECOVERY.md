# Recovery provenance

The research code was reconstructed on 2 September 2026 after the original
`zsgpu-5350` host was released. The local sources used were the base project,
`tmp/continuous_discovery_stage`, `tmp/continuous_resume_z8seo5`, the preserved
EHVI/turnover edits, `tmp/alpharesearch_diversity`, and development transcripts.
This provenance is not a claim that every lost remote edit was recovered.

## Current preserved evidence

The canonical R100 continuous runs for seeds 42, 123 and 456 are now versioned
inside `runs/ldm_continuous_discovery/`, including 342 ordered factors per seed,
resume-state exports, Top-5 reports and figures. The later five-round reporting
backfill has its own ledgers and checksum provenance. The single-factor library
and saved Stage II records are also retained under `runs/`. Partial records are
preserved as partial, not relabelled as completed experiments.

See the run README, `results_summary.json`, `legacy_import.json` where present,
and `five_round_checkpoint_import.json` for the evidence belonging to each run.
These are scientific records, not disposable caches.

## Rebuilt versus recovered

The official AlphaBench dependency and its audited integrity patch, local Conda
environment, and external Qlib provider were subsequently re-established. They
must not be described as a full recovered image of the old server. The exact
remote environment, complete runtime logs and uncaptured manual edits remain
unavailable.

An exported `resume_state.json` is not sufficient to guarantee an executable
resume: continuous discovery also checks its committed factor ledger, RNG and
search-signature contract. Preserve the distinction between being able to
report a saved run and being able to resume it exactly.

The active source of truth is the single `AlphaResearch` Git repository. Code
and exported experiment records should remain versioned; market data and
secrets require separate, deliberate backup rather than inclusion in Git.
