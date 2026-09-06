# AlphaResearch / AlphaLDM (recovered)

This repository reconstructs the final AlphaResearch architecture used for the
August 2026 AlphaLDM experiments. The primary method is
`ldm_continuous_discovery`: pure scalar-UCB AlphaLDM search with durable round
commits, ordered factor preservation, exact RNG-state capture, and staged
Validation-selected Top-5 Test audits.

## Scientific boundary

- Train metrics may guide generation, GP fitting, and acquisition.
- Validation is used only to select the reported factor or factor pool.
- Test is retrospective reporting only and never feeds back into search.
- The plotted Test best-so-far envelope is a retrospective diagnostic derived
  from the raw Test checkpoints; it is never used to select a factor or round.
- A continuous round is resumable only after its complete commit is present.

## Included methods

- `baseline_alpha158`
- `baseline_alphabench_cot`
- `ldm_standard`
- `ldm_cold_start`
- `ldm_prompt_harness`
- `ldm_minimal_prompt_harness`
- `ldm_single_factor`
- `ldm_split_robust_reward`
- `ldm_rankic_worst_qehvi`
- `ldm_rankic_rankicir_ehvi`
- `ldm_rankic_turnover_ehvi`
- `ldm_continuous_discovery`

## Installation

Python 3.10 or 3.11 is recommended on Linux.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The real evaluator also needs the official AlphaBench checkout at
`external/AlphaBench`. The experiments were pinned to commit
`31bb94bbb7744177c51c9d011e07c31e3092b93e`.

After cloning with submodules, restore the verified-evaluation FFO changes:

```bash
git submodule update --init --recursive
bash scripts/apply_alphabench_overlays.sh
```

LLM credentials are read only from the environment:

```bash
export ALPHARESEARCH_LLM_API_KEY='...'
```

Never put credentials in a YAML config or commit them to Git.

For the prepared Apple Silicon Conda environment, local Qlib/FFO setup, and
DeepSeek Flash launcher, see [`docs/LOCAL_MAC_SETUP.md`](docs/LOCAL_MAC_SETUP.md).

## Smoke test

```bash
pytest -q
python -m alpha_research.cli \
  configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml \
  --set args.mock=true \
  --set args.search-rounds=2 \
  --set args.ldm-random-seed=999 \
  --set args.continuous-checkpoint-round=[]
```

## Continuous discovery artifacts

Each run writes:

- `factor_ledger.jsonl`: append-only factor evaluations and round commits;
- `resume_state.json`: last safe round, RNG state, search signature, and GP size;
- `factor_sequence.json`: every saved factor in real evaluation order;
- `top5_combinations.json`: R0/R10/... checkpoint Validation Top-5 and measured
  Test audit;
- `events.jsonl`, `search.json`, `validation_selection.json`, and `summary.json`.

Use `scripts/run_ldm_continuous_discovery.sh TARGET_ROUNDS SEED` on the original
Linux layout, or invoke the CLI directly with environment-specific paths.

## Stage 2 Method 1: worst-quarter Train objective

`ldm_split_robust_reward` keeps the standard AlphaLDM proposal, 12D profile,
GP, kernel, acquisition, and evaluation budget. Its only search change is the
scalar GP target: one full 2016--2020 Train evaluation is grouped into the 20
calendar quarters, and `worst_rankic` is the minimum quarterly mean signed
RankIC. The run saves `train_rankic`, `worst_rankic`, `worst_quarter`, and all
20 quarterly values in `ldm_history.csv` and `split_reward_history.csv`.

## Stage 2 Method 2: RankIC-Worst qEHVI

`ldm_rankic_worst_qehvi` reuses Method 1's exact 20-quarter aggregation and
maximises `(train_rankic, worst_rankic)` in the Pareto sense. It fits two
independent GPs on the existing 12D factor representation. Per-objective
z-score normalisation and the dominated reference point are frozen from the 42
initial Alpha158 observations. With the default 8-proposal/3-evaluation round,
all 56 candidate triples are scored by Monte Carlo qEHVI using each GP's joint
candidate covariance, and the best triple is evaluated.

The 38-round DeepSeek Flash config is:

```bash
python scripts/run_experiment.py \
  configs/ldm_rankic_worst_qehvi/ldm_rankic_worst_qehvi_neolink-deepseek-v4-flash_2016-2025.yaml
```

In addition to the standard artifacts, a run writes `pareto_archive.json` and
`mobo_state.json`; these record the raw objectives, all quarterly RankICs,
current Pareto membership, frozen normalisation/reference point, and final
normalised hypervolume. Validation retains the standard RankIC Top-30 selection
under the 0.8 daily-RankIC correlation boundary, and Test remains report-only.

To backfill the denser R0/R10/.../R100 reporting schedule from preserved factor
sequences without rerunning discovery or calling the LLM, start the real FFO
evaluator and run:

```bash
python scripts/backfill_ldm_checkpoint_reports.py --seeds 42 123 456
```

Use `--dry-run` first to list missing checkpoints. Backfill evaluates and
selects on Validation, audits the frozen Top-5 on Test, writes a separate
`checkpoint_backfill_events.jsonl`, and never changes search state.

## Single-factor library

Build a cross-seed library of LDM-generated factors whose Train RankIC is at
least 0.04:

```bash
python scripts/build_single_factor_library.py --seeds 42 123 456
```

The complete candidate table and the strict Validation-admitted view are
written under `runs/ldm_continuous_discovery/factor_library/`.  Formula,
explanation, seed, round and original evaluation index are retained, and
canonical expressions are de-duplicated across seeds.  A high Train value only
creates a candidate; Validation RankIC >= 0.04 is required for admission. Test
is audit-only and never changes library membership.

If restored market data and the FFO service are available, evaluate only the
currently missing candidate Validation results (no LLM and no search rerun):

```bash
python scripts/backfill_single_factor_validation.py --dry-run
python scripts/backfill_single_factor_validation.py
```

The append-only Validation ledger makes this job safely resumable.

To measure every Train-qualified candidate on Test for a one-time descriptive
audit, without changing formal library admission, run:

```bash
python scripts/audit_single_factor_test.py --dry-run
python scripts/audit_single_factor_test.py
```

This writes a separate append-only Test ledger and
`test_audit_over_threshold.*`.  These files are retrospective analysis only;
they must not be presented as Validation-selected performance.

## Preserved R100 experiments

The completed continual-discovery result snapshots for seeds 42, 123, and 456
are versioned with this repository under `runs/ldm_continuous_discovery/`, using
the same canonical run names as the original GPU experiments. Each run directory
contains its ordered factor sequence, resume state, checkpoint Top-5
Validation/Test reports, and PNG/SVG/PDF figure. See the run directory's README
and `results_summary.json` for a compact index.

## Recovery provenance

The reconstruction merged the last locally preserved code snapshots in this
order: the AlphaResearch base, final continual-discovery stage, final legacy
resume patch, RankIC/RankICIR EHVI, RankIC/turnover EHVI, and prompt-diversity
modules. See `RECOVERY.md` for boundaries and known missing external artifacts.
