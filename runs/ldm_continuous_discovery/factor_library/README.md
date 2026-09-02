# AlphaLDM single-factor library

This directory is derived from the ordered `factor_sequence.json` artifacts.

- Candidate gate: `Train RankIC >= 0.04`
- Admission gate: `Validation RankIC >= 0.04`
- Test: retrospective audit only; never used for admission
- Unique Train candidates: 126
- Validation-admitted factors: 79
- Pending Validation: 0
- Below the Validation threshold: 47
- Measured on Test: 126
- Test audit >= 0.04: 7
- Train/Validation/Test all above threshold: 4

`single_factor_library.*` contains every Train candidate and its status.
`admitted_factors.*` is the strict factor-library view.  Formula, explanation,
seed, round and original evaluation order are retained.  Test values, when
present, are audit metadata and cannot admit a factor.
`test_audit_over_threshold.*` is a descriptive retrospective view, not a
selection or admission result.
`three_split_audit_over_threshold.*` contains the post-hoc intersection of the
Train, Validation and Test thresholds and carries the same Test-audit warning.
