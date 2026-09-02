"""Minimal, independent hypothesis-first proposal contract for AlphaLDM."""

from .generator import MinimalHarnessCandidateGenerator
from .method import MinimalHarnessLDM

__all__ = ["MinimalHarnessCandidateGenerator", "MinimalHarnessLDM"]
