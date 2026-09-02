# AlphaLDM single-factor library

This directory is derived from the ordered `factor_sequence.json` artifacts.

- Candidate gate: `Train RankIC >= 0.04`
- Admission gate: `Validation RankIC >= 0.04`
- Test: retrospective audit only; never used for admission
- Unique Train candidates: 126
- Validation-admitted factors: 18
- Pending Validation: 96
- Below the Validation threshold: 12

`single_factor_library.*` contains every Train candidate and its status.
`admitted_factors.*` is the strict factor-library view.  Formula, explanation,
seed, round and original evaluation order are retained.  Test values, when
present, are audit metadata and cannot admit a factor.
