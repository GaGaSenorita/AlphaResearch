"""AlphaLDM variant that changes only the single-candidate prompt contract."""

from alpha_research.methods.ldm.method import AlphaLDM


class MinimalHarnessLDM(AlphaLDM):
    name = "ldm_minimal_prompt_harness"
    candidate_source = "ldm_minimal_prompt_harness"
