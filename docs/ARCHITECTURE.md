# Research architecture

The CLI expands YAML, `arguments.py` defines options, and `runner.py` builds the
method, profiler and evaluator. `StaticEnvironment` owns the chronological
Train/search, Validation/selection and test/reporting protocol.

## Method map

| Role | Modules / configured method names |
| --- | --- |
| Shared LDM search | `methods/ldm`: proposal, history, GP, acquisition and scalar search |
| Pure LDM | `ldm_standard`; `ldm_continuous_discovery` adds durable round commits |
| Stage II Method 1 | `ldm_split_robust_reward`, using annual `split_robust/reward.py` |
| Stage II Method 2 | `ldm_rankic_worst_qehvi`, sharing annual reward and `ldm/multi_objective.py` |
| Baselines | `baseline_alpha158`, `baseline_alphabench_cot` |
| Other variants | `ldm_cold_start`, `ldm_single_factor`, `ldm_prompt_harness`, `ldm_minimal_prompt_harness`, `ldm_rankic_rankicir_ehvi`, `ldm_rankic_turnover_ehvi` |

All original method identifiers remain. `environments/online.py` is a rolling-
protocol library; the research CLI currently enables only the static environment.
The API dashboard belongs to a separate branch.

## Pure LDM data flow

1. Load the fixed Alpha158 seeds; evaluate on Train and compute label-free profiles.
2. Build LLM context from the training incumbent and recent verified training records.
3. Generate, validate, deduplicate and profile candidate expressions.
4. Fit the GP on verified profile/reward pairs; select candidates through UCB and
   pool-standardised sampling.
5. Verify external formula identity, period and finite daily measurements before
   appending observations. Valid negative rewards remain in the archive.
6. Update the context and commit complete rounds. Run configured Validation/test
   reports separately from search.

`formula.py` and `canonical.py` implement DSL checks. `profile.py` provides a fixed
12-dimensional map within Train. `evaluator.py` independently verifies the external
service. Vendor changes are confined to `integrations/alphabench/`.

## Integrity and continuation

Explicit profiling bounds must be paired and contained in Train. A complete GP
refit initialises every parameter, including likelihood noise, from the same
configuration. Resume checks cover search settings, profiling identity and the
GP fitting policy; incompatible historical signatures are rejected.

Continuous discovery protects existing output directories, verifies the ledger
and can recover a torn final append while retaining those bytes for inspection.
Middle-of-file corruption or altered measurements are errors. Failed final test
evaluations cannot be reported as completed successful experiments.

These properties are tested with controlled synthetic evaluations. Historical
financial records have not been regenerated under the cleanup's corrected code.

## Reporting and entry points

Plotters read saved records without reselecting factors or requesting evaluation.
`reporting/output.py` saves one visual format (PDF by default); JSON/CSV retain
measurements and provenance.

| Entry point | Purpose |
| --- | --- |
| `run_experiment.py` | Configuration-first experiment CLI |
| `run_stage2_seed_matrix.py` | Annual Stage II queue and compatible-result checks |
| `setup_alphabench.py` | Pinned dependency and integrity patch verification |
| `render_ldm_report_figures.py` | Three LDM run plots and combined comparison |
| `render_stage1_thesis_figures.py` | Thesis trajectories from saved checkpoints |
| `plot_representative_factor_nav.py` | Factor-sorted NAVs from daily CSVs |
| `backfill_ldm_checkpoint_reports.py` | Verified Validation/test backfill without search |
| `build_single_factor_library.py` | Cross-seed factor library from saved records |
| `backfill_single_factor_validation.py`, `audit_single_factor_test.py` | Missing real measurements |
| `reanalyse_validation_worst_year_topk.py` | Explicit post-hoc annual selection analysis |
| `package_submission.py` | Source-and-evidence ZIP with checksums |

Backfill and audit commands requesting missing measurements need the market data
and evaluator. They are distinct from saved-data plotting.
