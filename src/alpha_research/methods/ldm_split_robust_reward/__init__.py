"""AlphaLDM with the Stage-2 Method-1 worst-quarter Train objective."""

from alpha_research.methods.split_robust.reward import SplitSettings
from .method import SplitRobustLDM

__all__ = ["SplitRobustLDM", "SplitSettings"]
