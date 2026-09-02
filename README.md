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
- `top5_combinations.json`: checkpoint Validation Top-5 and measured Test audit;
- `events.jsonl`, `search.json`, `validation_selection.json`, and `summary.json`.

Use `scripts/run_ldm_continuous_discovery.sh TARGET_ROUNDS SEED` on the original
Linux layout, or invoke the CLI directly with environment-specific paths.

## Preserved R100 experiments

The completed continual-discovery result snapshots for seeds 42, 123, and 456
are versioned with this repository under
`results/ldm_continuous_discovery/`. Each seed directory contains its ordered
factor sequence, resume state, checkpoint Top-5 Validation/Test reports, and
PNG/SVG/PDF figure. See the result directory's README and `summary.json` for a
compact index.

## Recovery provenance

The reconstruction merged the last locally preserved code snapshots in this
order: the AlphaResearch base, final continual-discovery stage, final legacy
resume patch, RankIC/RankICIR EHVI, RankIC/turnover EHVI, and prompt-diversity
modules. See `RECOVERY.md` for boundaries and known missing external artifacts.
