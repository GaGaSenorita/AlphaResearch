# Research architecture

The CLI expands YAML once, `arguments.py` defines the option contract, and
`runner.py` builds the method, profiler and evaluator. `StaticEnvironment` owns
the Train/search → Validation selection → frozen Test reporting protocol.
`runner.parse_args` remains available for existing integrations.

## Methods

| Role | Modules / configured method names |
| --- | --- |
| Shared AlphaLDM search | `methods/ldm`: proposal, history, GP, acquisition and scalar search loop |
| Stage I baseline | `ldm_standard`; `ldm_continuous_discovery` adds durable round commits and checkpoint reports |
| Stage II Method 1 | `ldm_split_robust_reward`, using `split_robust/reward.py` |
| Stage II Method 2 | `ldm_rankic_worst_qehvi`, sharing the quarterly reward and `ldm/multi_objective.py` |
| Baselines | `baseline_alpha158`, `baseline_alphabench_cot` |
| Configured research variants | `ldm_cold_start`, `ldm_single_factor`, `ldm_prompt_harness`, `ldm_minimal_prompt_harness`, `ldm_rankic_rankicir_ehvi`, `ldm_rankic_turnover_ehvi` |

Variants are retained because they have explicit configurations and/or tests;
being absent from the main paper does not make them disposable. This map does
not claim that every seed or variant has a completed experiment.

`environments/online.py` is a retained rolling-protocol library, not the API
dashboard's real-time execution mode. The research CLI currently accepts only
the static environment. The FastAPI application lives on `AlphaLDM_API`.

## Shared services

- `formula.py` / `canonical.py`: DSL validation and canonical deduplication.
- `profile.py`: fixed-window, label-free 12D representation.
- `evaluator.py` / `alphabench_runtime.py`: verified external results and pinned
  AlphaBench loading. Vendor deltas live in `integrations/alphabench/`.
- `factor_library.py`: cross-seed library admission and descriptive audits.
- `reporting/data.py`: pure saved-record extraction; no live evaluation.
- `reporting/financial.py` / `comparison.py`: individual and cross-seed plots.

Reporting never recomputes Validation selection inside the plotter. It reads the
saved Top-5 audits, checks the requested measurement grid, and rejects missing
or non-finite Test points. Manual projections are not supported.

## Script responsibilities

| Entry | Responsibility |
| --- | --- |
| `run_experiment.py` | Configuration-first experiment CLI |
| `run_local_continuous_discovery.sh` | Prepared macOS runtime launcher |
| `run_stage2_seed_matrix.py` | Existing bounded Stage II seed queue |
| `setup_alphabench.py`, `check_local_setup.sh` | Integration and runtime checks |
| `render_ldm_report_figures.py` | Three individual exports, run copies and comparison |
| `plot_ldm_financial_mining.py`, `plot_ldm_three_seed_comparison.py` | Thin individual plotting entry points |
| `backfill_ldm_checkpoint_reports.py` | Real Validation/Test reporting backfill; no search rerun |
| `build_single_factor_library.py` | Saved-record factor library construction |
| `backfill_single_factor_validation.py`, `audit_single_factor_test.py` | Missing real single-factor measurements |

Backfill and experiment commands use the same `cli.resolved_runner_args`
expansion. The scripts do not maintain separate parameter defaults.

## Cleanup boundary (September 2026)

The cleanup removes the old 5350-specific shell launchers, the one-off
`run_ldm_legacy_pair.py` worker, unused historical `methods/alphaldm` and
`methods/baselines` import aliases, and obsolete poster/single-factor/projection
plot layouts. Their previous contents are recoverable from Git commit
`7842fd9`. Verified legacy import logic remains in
`methods/ldm_continuous_discovery/legacy.py` and keeps its tests and CLI option.

No method reward, proposal budget, GP setting, canonical config/run name, stored
result, factor library or saved scientific figure is changed. The API branch
is separate and is not merged or reset by this cleanup.

The synthetic `MockFactorEvaluator` now emits deterministic ISO-dated weekdays
across the requested period instead of five malformed dates. This repairs
quarterly-objective smoke tests; it does not change real evaluation. Mock runs
created before this change should not be resumed with the new synthetic data.
