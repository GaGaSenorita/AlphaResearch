# AlphaResearch / AlphaLDM

AlphaLDM searches for formulaic financial factors using LLM proposals, a
label-free 12D representation, Gaussian-process surrogates and verified external
evaluation. This repository contains the research implementation and preserved
experiment results. Recovery provenance is documented in [docs/RECOVERY.md](docs/RECOVERY.md).

## One repository, two branches

- `main`: research methods, experiment configurations, saved results and figures.
- `AlphaLDM_API`: FastAPI backend and mock/online dashboard.

Use **one `AlphaResearch` directory**. Switch with `git switch main` or
`git switch AlphaLDM_API`; no second checkout is needed. Stop workers and the API
server before switching, and commit or stash local edits first. Ignored local
environments and `runtime/` records remain on disk. API installation and startup
instructions live on the API branch; `main` does not serve the dashboard.

## Install

The dependency source is `pyproject.toml`; `environment.yml` installs it into the
Conda environment `AlphaResearch` rather than duplicating a requirements list.
Python 3.11 is the prepared local runtime.

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate AlphaResearch
python scripts/setup_alphabench.py --apply
```

For an existing environment, use `conda env update -f environment.yml`.
Alternatively install with `pip install -e '.[dev,ffo]' -e external/AlphaBench`.
The setup command verifies the pinned AlphaBench revision and installs only the
audited integrity patch. It refuses to overwrite local changes. Running it
without `--apply` only inspects the integration.

Real evaluation additionally needs the Qlib CSI300 data provider and an FFO
service. Real proposal generation needs `ALPHARESEARCH_LLM_API_KEY` in the
environment. Neither market data nor credentials are included in Git.
See [local setup](docs/LOCAL_MAC_SETUP.md) and
[the evaluator integration](integrations/alphabench/README.md).

## Entry points

```bash
alpha-research --list
alpha-research configs/ldm_continuous_discovery/ldm_continuous_discovery_neolink-deepseek-v4-flash_2016-2025.yaml --dry-run
```

`python -m alpha_research.cli` and `python scripts/run_experiment.py` expose
the same configuration-first entry point. `--dry-run` validates the protocol
and paths without running an experiment or calling the LLM.

On the prepared Mac, start real continual discovery with
`bash scripts/run_local_continuous_discovery.sh 100 42`. The first argument is
the cumulative target, not the number of additional rounds. Only committed
rounds are resumable; an exported result snapshot alone is not a complete
executable checkpoint.

To redraw the saved three-seed results without data, an evaluator or API calls:

```bash
python scripts/render_ldm_report_figures.py
```

## Project layout

```text
configs/                 Experiment configurations; canonical names are stable
src/alpha_research/
  cli.py, arguments.py   Config expansion and experiment option contract
  runner.py             Component construction and experiment orchestration
  environments/         Evaluation protocols
  methods/              AlphaLDM, baselines and research variants
  evaluator.py          Verified evaluation and result validation
  profile.py            Label-free behavioural representation
  reporting/            Saved-data extraction and scientific plotting
scripts/                 Launch, reporting, audit and setup entry points
integrations/alphabench/ Pinned, checksummed FFO patch (no duplicate vendor code)
external/AlphaBench/     Official Git submodule
runs/                    Preserved evidence and local experiment outputs
figures/                 Exported scientific figures
tests/                   Automated correctness and regression checks
docs/                    Setup, methods, architecture and provenance
```

`runtime/` and local environments are ignored and are not source modules.
Existing runtime jobs and recovered results should not be removed as build
artifacts. `runs/` keeps the original method/model/period/seed naming scheme.

## Research methods and scientific boundaries

The principal configurations are continuous scalar-UCB AlphaLDM,
`ldm_split_robust_reward` (worst quarterly Train RankIC), and
`ldm_rankic_worst_qehvi` (Train RankIC–worst-quarter batch qEHVI).
Baselines and configured ablations remain available; see
[the method map](docs/ARCHITECTURE.md) and [experiment guide](docs/EXPERIMENTS.md).

Train guides search. Validation selects and filters factors for reporting.
Test is report-only and never feeds into either process. The displayed Test
best-so-far envelope is explicitly retrospective, with the raw measured
checkpoints retained. Preserved R100 runs have the complete R0/R5/.../R100
reporting grid; a new run follows its chosen configuration, not the figure grid.

## Verify

```bash
MPLBACKEND=Agg pytest -q
```

Tests use synthetic evaluations and preserved records, not paid LLM calls.
They cover verified results, round continuation, quarterly rewards, MOBO,
configurations, patch safety and five-round scientific reporting.
