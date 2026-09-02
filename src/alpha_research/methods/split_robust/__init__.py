"""Train-only temporal robustness scoring."""

from .reward import SplitScore, SplitSettings, split_score

__all__ = ["SplitScore", "SplitSettings", "split_score"]
