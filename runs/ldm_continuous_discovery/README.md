# LDM continual-discovery R100 results

These are the locally preserved result snapshots from the three completed
`ldm_continuous_discovery` experiments. They are kept inside the repository so
the code, generated factors, measured effects, and presentation figures travel
together.

| Seed | Completed round | Ordered factors | Final Top-5 Test RankIC | Best checkpoint Test RankIC |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 100 | 342 | 0.04243189 | 0.04243189 (R100) |
| 123 | 100 | 342 | 0.04495599 | 0.04528040 (R38) |
| 456 | 100 | 342 | 0.03361301 | 0.03836634 (R38/R50/R75) |

Each canonical `ldm_continuous_discovery_*_seed<seed>/` directory contains:

- `factor_sequence.json`: all 342 factors in committed real-evaluation order,
  including each factor's expression and Train metrics;
- `resume_state.json`: R100 safe state, search signature, GP observation count,
  LLM call count, and encoded RNG state;
- `top5_combinations.json`: R20/R38/R50/R75/R100 Validation-selected Top-5
  combinations with measured Validation/Test reports and daily metrics;
- `figure.{png,svg,pdf}` and `figure_metadata.json`: the corresponding two-panel
  Train/Test result figure and plotting inputs;
- `legacy_import.json` for seeds 123 and 456, documenting migration of their
  original 38-round histories;
- `checkpoint_validation_ledger.jsonl` for seed 42, which was copied with its
  R100 snapshot.

Test metrics are retrospective audits only. They were not used to guide search
or select factors.

## Recovery boundary

These directories contain the core R100 scientific results that were copied to
the local machine before the original GPU host was reinstalled. They are not a
byte-for-byte copy of the lost remote `runs/` directories: full runtime logs,
all event streams, and the complete append-only factor ledger for every seed
were not preserved. The saved artifacts are sufficient to inspect every
generated factor, recover its Train metrics, reproduce the existing figures,
and audit every staged Top-5 Validation/Test result.
