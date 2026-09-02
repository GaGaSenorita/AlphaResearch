"""AlphaLDM with a hypothesis-first, batch-aware candidate prompt harness."""

from alpha_research.methods.ldm.method import AlphaLDM


class HarnessLDM(AlphaLDM):
    """Reuse AlphaLDM unchanged after replacing only candidate generation."""

    name = "ldm_prompt_harness"
    candidate_source = "ldm_prompt_harness"
