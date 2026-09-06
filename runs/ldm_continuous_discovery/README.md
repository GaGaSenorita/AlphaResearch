# LDM continual-discovery R100 results

These are the locally preserved result snapshots from the three completed
`ldm_continuous_discovery` experiments. They are kept inside the repository so
the code, generated factors, measured effects, and presentation figures travel
together.

| Seed | Completed round | Ordered factors | Final Top-5 Test RankIC | Best five-round checkpoint Test RankIC (first occurrence) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 100 | 342 | 0.04243189 | 0.04243190 (R90) |
| 123 | 100 | 342 | 0.04495599 | 0.04863754 (R5) |
| 456 | 100 | 342 | 0.03361301 | 0.03836634 (R25) |

Each canonical `ldm_continuous_discovery_*_seed<seed>/` directory contains:

- `factor_sequence.json`: all 342 factors in committed real-evaluation order,
  including each factor's expression and Train metrics;
- `resume_state.json`: R100 safe state, search signature, GP observation count,
  LLM call count, and encoded RNG state;
- `top5_combinations.json`: measured Validation-selected Top-5 combinations at
  all 21 checkpoints R0/R5/.../R100, plus the preserved off-grid R38 audit.
  Original records are unchanged; additional checkpoints are post-hoc reports
  using only the factors available at that round, not new discovery runs;
- `figure.{png,svg,pdf}` and `figure.json`: the corresponding two-panel
  Train/Test result figure and plotting inputs. The gray points are raw Test
  audits and the teal step is the explicitly retrospective best-so-far envelope;
- `legacy_import.json` for seeds 123 and 456, documenting migration of their
  original 38-round histories;
- `checkpoint_validation_ledger.jsonl`: the original Validation ledger plus
  append-only reporting backfill records for each seed;
- `five_round_checkpoint_import.json`: checksums and provenance of the imported
  five-round reports, including checks that search state, factor order and all
  existing checkpoint records were preserved;
- `checkpoint_measurement_provenance.jsonl.gz`: evidence of real measurements
  and exact same-pool measurement reuse during the reporting backfill.

Test metrics are retrospective audits only. They were not used to guide search
or select factors. The best-so-far curve depends on the checkpoints included;
it is not the final R100 pool score. Differences at roughly 1e-8 precision,
such as seed 42's R90 versus R100 values, are not meaningful improvements.

Run `python scripts/render_ldm_report_figures.py` from the repository root to
regenerate all four figures and the per-run copies without any API calls.

## Recovery boundary

These directories contain the core R100 scientific results that were copied to
the local machine before the original GPU host was reinstalled. They are not a
byte-for-byte copy of the lost remote `runs/` directories: full runtime logs,
all event streams, and the complete append-only factor ledger for every seed
were not preserved. The saved artifacts are sufficient to inspect every
generated factor, recover its Train metrics, reproduce the existing figures,
and audit every staged Top-5 Validation/Test result.
