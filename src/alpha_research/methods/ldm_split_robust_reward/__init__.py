"""AlphaLDM with a train-only cross-year stability objective."""

from alpha_research.methods.split_robust.reward import SplitSettings
from .method import SplitRobustLDM

__all__ = ["SplitRobustLDM", "SplitSettings"]
