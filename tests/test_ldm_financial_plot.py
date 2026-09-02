from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_ldm_financial_mining.py"
SPEC = importlib.util.spec_from_file_location("plot_ldm_financial_mining", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def test_retrospective_test_best_so_far_keeps_raw_values_and_source_rounds():
    records = PLOT.retrospective_test_best_so_far(
        [0, 10, 20, 30],
        [0.030, 0.040, 0.035, 0.041],
    )
    assert [row["rank_ic"] for row in records] == [0.030, 0.040, 0.040, 0.041]
    assert [row["raw_rank_ic"] for row in records] == [0.030, 0.040, 0.035, 0.041]
    assert [row["source_round"] for row in records] == [0, 10, 10, 30]
    assert [row["improved"] for row in records] == [True, True, False, True]
