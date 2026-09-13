"""AlphaLDM with the Stage-2 Method-1 worst-year Train objective."""

from alpha_research.methods.split_robust.reward import YearlySplitSettings
from .method import SplitRobustLDM

SplitSettings = YearlySplitSettings

__all__ = ["SplitRobustLDM", "SplitSettings", "YearlySplitSettings"]
