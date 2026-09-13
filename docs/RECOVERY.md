# Recovery and result provenance

The research implementation was reconstructed in September 2026 after the original
GPU host was released. Sources included preserved project files, intermediate
checkouts and development records. This is not a complete image of that server.

The canonical pure LDM R100 snapshots for seeds 42, 123 and 456 preserve 342
ordered factors each, training metrics and measured Validation/test combinations.
Later five-round reporting backfills have separate ledgers and checksums. See
`runs/ldm_continuous_discovery/README.md`, `results_summary.json`, each available
`legacy_import.json`, and `five_round_checkpoint_import.json`.

Full original runtime logs and all append-only factor ledgers were not recovered.
The snapshots reproduce the saved reports and figures; they do not guarantee exact
executable continuation. The current code also tightens profiling/result checks
and resets every GP parameter during a full refit, including likelihood noise.
Historical signatures are deliberately incompatible with that changed fit policy.
Legacy continuation requires a matching recorded GP fit policy, except for
verified historical runs that never trained GP hyperparameters.
Stored scientific results were not regenerated or relabelled during cleanup.

Stage II records were relocated to a local-only sibling directory with per-file
SHA256 verification. Their original partition and completion metadata remain
unchanged. Archived quarterly runs are distinct from current annual methods.
See `docs/DATA_LAYOUT.md` for paths and the local migration manifest.

The official AlphaBench revision and integrity patch are pinned and checksummed.
The Qlib provider and API credentials remain external to the submission. Third-party
code is identified separately from AlphaResearch source.
