# AlphaResearch / AlphaLDM

AlphaLDM adapts the Large Discovery Model framework to formulaic alpha mining.
It combines LLM proposals, label-free behavioural representations, Gaussian-process
surrogates and externally verified factor evaluation.

This submission preserves all research method implementations, configurations and
tests. The experiment data on `main` are the three pure LDM runs and their derived
factor library and figures. Stage II results remain local-only; both Stage II
methods now use **calendar-year** training splits.

## Methods

| Research role | Method | Search objective |
| --- | --- | --- |
| Pure LDM | `ldm_standard` | Mean daily signed training RankIC; GP/UCB selection |
| Resumable pure LDM | `ldm_continuous_discovery` | Same objective, with verified round commits |
| Stage II Method 1 | `ldm_split_robust_reward` | Minimum yearly mean training RankIC |
| Stage II Method 2 | `ldm_rankic_worst_qehvi` | Joint mean training RankIC and worst-year RankIC; batch qEHVI |

Baseline Alpha158, AlphaBench CoT, cold-start, single-factor, prompt-harness,
RankIC/RankICIR and RankIC/turnover variants remain available. See the complete
[method map](docs/ARCHITECTURE.md). Code availability does not imply a completed
experiment for every variant.

## Installation

Use Python 3.11 and run commands from this directory. For a Git clone:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ffo]' -e external/AlphaBench
python scripts/setup_alphabench.py --apply
```

Keep the editable (`-e`) installation: commands locate bundled configurations,
integrations and saved results relative to the source directory.

The submission ZIP includes pinned AlphaBench source, so omit the `git submodule`
command when using it. The integration checker supports Git checkouts and
checksummed source exports. See [the integration guide](integrations/alphabench/README.md)
for attribution and patch details.

Alternatively use `conda env create -f environment.yml`. Tested direct dependency
versions are recorded in [docs/TESTED_ENVIRONMENT.md](docs/TESTED_ENVIRONMENT.md).

## Verify without API calls or market data

```bash
alpha-research --list
MPLBACKEND=Agg python -m pytest -q
python scripts/run_experiment.py configs/smoke/ldm_continuous_discovery_mock_2016-2025.yaml
```

The smoke configuration uses synthetic generation, profiling and evaluation.
It writes to ignored `runtime/smoke/` and is a software check, not a financial
experiment. Re-running it resumes the same completed smoke run.

To inspect a real configuration without starting an experiment:

```bash
alpha-research configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml --dry-run
```

Real runs additionally need a Qlib CSI300 provider, a running AlphaBench FFO
service and the API-key environment variable specified in the chosen config.
Market data and credentials are excluded. Set `args.qlib-provider-uri` and
`args.ffo-url` for your installation; see [experiments](docs/EXPERIMENTS.md) and
[optional macOS setup](docs/LOCAL_MAC_SETUP.md). Real generation incurs API usage.

## Reproduce the saved LDM figures

```bash
python scripts/render_ldm_report_figures.py
python scripts/render_stage1_thesis_figures.py
python scripts/plot_representative_factor_nav.py
```

These commands read preserved records or daily CSVs without calling the LLM or
factor evaluator. Each plot has one canonical **PDF** in `figures/`; JSON/CSV
preserve inputs and provenance. Use `--format png` or `--format svg` for an
optional local export. See the [NAV guide](docs/REPRESENTATIVE_FACTOR_NAV.md).

## Data and interpretation

Train alone guides generation, surrogate fitting and acquisition. Validation
selects factors and checks correlation; test data evaluate the fixed combination.
Test values never feed back into search. Displayed test best-so-far curves are
explicitly retrospective, not deployable selection rules.

The preserved LDM runs have 342 factors each: 42 initial factors plus 100 rounds
of three evaluations, for seeds 42, 123 and 456. Recorded measurements are unchanged.
These are reporting snapshots, not complete executable checkpoints under the
revised integrity checks; see [provenance](docs/RECOVERY.md).

Stage II outputs default to `../AlphaResearch_local_results/stage2/`. Old quarterly
records are separate from annual runs and must not be relabelled as annual
experiments. See [data organisation](docs/DATA_LAYOUT.md).

## Layout and submission

```text
configs/                  Method configurations and mock smoke test
src/alpha_research/        Search, evaluation, profiling and reporting
scripts/                  Experiment, plotting, audit and packaging tools
tests/                    Correctness and regression tests
integrations/alphabench/   Pinned evaluator patch and checksums
external/AlphaBench/      Third-party submodule or verified source export
runs/ldm_continuous_discovery/  Preserved pure LDM evidence
figures/                  Canonical PDFs and their data/provenance
docs/                     Architecture and reproducibility
```

Runtime files, environments and local results are excluded from Git and the
submission archive. The separate `AlphaLDM_API` branch contains dashboard work
and is not required for this research submission.

Build a source-and-evidence archive without Git history or local runtime files:

```bash
python scripts/package_submission.py --output ../AlphaResearch_submission.zip
```
