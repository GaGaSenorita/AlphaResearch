# Experiment and reporting guide

Run from the repository root after installation. Real search calls the configured
LLM; use the dedicated smoke config for a fully synthetic, offline check.

## Pure LDM

```bash
python scripts/run_experiment.py configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml --dry-run
```

Remove `--dry-run` only when real evaluation and generation are intended. Override
settings with `--set args.NAME=VALUE`, for example
`--set args.qlib-provider-uri=/path/to/cn_data` or
`--set args.llm-api-key-env=MY_LLM_API_KEY`. Credentials themselves stay in the
environment, never in command arguments or YAML.

`search-rounds` is the cumulative target for continuous discovery. New runs use
canonical method/model/period/seed names. Use a new seed or output parent for a
new experiment. Exact continuation requires the matching committed factor ledger,
RNG state, configuration and profiling identity; a saved result snapshot alone is
insufficient. The current integrity checks reject incompatible old signatures.

A continuous run records `factor_ledger.jsonl`, `resume_state.json`,
`factor_sequence.json`, `top5_combinations.json`, protocol, events and summaries.
Only verified training observations update search; configured Validation/test
reports remain outside the search history. An incomplete round is not committed.

## Stage II: annual training splits

Both methods partition the full Train evaluation into five calendar years,
2016 through 2020. Let each year's score be its mean daily signed RankIC.
Method 1 optimises the minimum of those five means. Method 2 optimises that
minimum jointly with the full-Train mean, which weights valid days equally rather
than giving equal weight to years with different numbers of observations.

Missing or invalid yearly observations fail evaluation; rewards are not sign-
flipped or replaced with zero. Both methods retain the standard behaviour map and
verified evaluation path. Method 2 fits two independent GPs and uses frozen
initial-objective normalisation/reference points with Monte Carlo batch qEHVI.

```bash
python scripts/run_experiment.py configs/ldm_split_robust_reward/ldm_split_robust_reward_deepseek-v4-pro_2016-2025.yaml --dry-run
python scripts/run_experiment.py configs/ldm_rankic_worst_qehvi/ldm_rankic_worst_qehvi_neolink-deepseek-v4-flash_2016-2025.yaml --dry-run
python scripts/run_stage2_seed_matrix.py --help
```

The configs retain their explicit provider/model settings; a controlled comparison
must align these settings as well as budgets and seeds. Do not infer a matched
comparison solely from method names.

Results default to `../AlphaResearch_local_results/stage2/<method>/<run_name>`.
They are excluded from the research submission. Historical quarterly outputs are
archived separately and refused as annual output destinations. An annual summary
of an old daily series does not turn a quarterly-guided search into an annual run.

Each run records annual reward history, yearly valid-day counts and protocol
metadata. Method 2 also writes `pareto_archive.json` and `mobo_state.json` using
the annual schema. Validation selection uses signed mean RankIC and the configured
correlation/combination rules; test results never guide either method.

## Saved-data plotting

```bash
python scripts/render_ldm_report_figures.py
python scripts/render_stage1_thesis_figures.py
python scripts/plot_representative_factor_nav.py
```

One PDF per figure is written under `figures/`, with JSON/CSV provenance. The
preserved R100 runs have all 21 measured checkpoints R0/R5/.../R100. Plotters reject
missing/non-finite points; they neither interpolate results nor select checkpoints
using test performance. The plotted best-so-far envelope is explicitly retrospective.

## Additional measurements and factor library

The commands below may request real evaluation. Review their `--dry-run` plans
first; they do not call the LLM or rerun discovery:

```bash
python scripts/backfill_ldm_checkpoint_reports.py --dry-run
python scripts/backfill_single_factor_validation.py --dry-run
python scripts/audit_single_factor_test.py --dry-run
```

`python scripts/build_single_factor_library.py` only reads saved records. Train
RankIC >= 0.04 creates a candidate, and Validation RankIC >= 0.04 is required for
library admission. Test audits never change membership. Full tables and the
Validation-admitted view remain in the preserved LDM factor library.

`reanalyse_validation_worst_year_topk.py` is an explicitly post-hoc analysis tool.
With Validation restricted to the single year 2021, its worst-year and mean scores
are identical; it cannot supply an additional temporal-robustness criterion.
