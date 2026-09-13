"""Save one figure format per plot; retain numerical sidecars separately."""
from __future__ import annotations

from pathlib import Path


FIGURE_FORMATS = ("pdf", "png", "svg")


def save_figure(fig, prefix: Path, *, output_format: str = "pdf", dpi: int = 240,
                **kwargs) -> Path:
    """Save a publication PDF by default, or one explicitly requested format."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(f"Unsupported figure format: {output_format}")
    path = prefix.with_suffix("." + output_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, **kwargs)
    return path
