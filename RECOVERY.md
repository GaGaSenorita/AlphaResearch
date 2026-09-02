# Recovery record

Recovered on 2026-09-02 after the original `zsgpu-5350` host was released.

## Sources used

1. Local `AlphaResearch` base project.
2. `tmp/continuous_discovery_stage` (2026-08-30).
3. `tmp/continuous_resume_z8seo5` (2026-09-01).
4. `.codex_ehvi_edit` and `.codex_turnover_edit.mg7z1G`.
5. `tmp/alpharesearch_diversity` and preserved development transcripts.

The old local project was not modified. This repository is a clean recovery
target and should become the new version-controlled source of truth.

## Preserved experiment evidence outside this repository

The parent workspace contains the R100 ordered factor state for seeds 42, 123,
and 456 under `data/ldm_financial_*_round100/`, plus their rendered figures.
Each factor sequence contains 342 entries.

## Not recovered from the released host

- the full remote `runs/` tree and runtime logs;
- the full Qlib market-data provider;
- the original AlphaBench checkout and FFO runtime;
- the exact Conda environment;
- any remote-only manual edit that was never captured by a development task.

R100 result artifacts are preserved, but continuing those exact runs requires
rebuilding a compatible `factor_ledger.jsonl` from the ordered factor state or
recovering the original ledger.
