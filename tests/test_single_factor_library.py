from __future__ import annotations

import json
from pathlib import Path

from alpha_research.factor_library import (
    build_single_factor_library,
    write_single_factor_library,
)


def _result(rank_ic: float, start: str, end: str) -> dict:
    return {
        "success": True,
        "expression": "$close",
        "period": {"start": start, "end": end},
        "metrics": {
            "rank_ic": rank_ic,
            "ic": rank_ic / 2,
            "rank_icir": 0.5,
            "icir": 0.25,
            "quantile_spread": None,
            "turnover": 0.7,
            "daily_count": 10,
            "observation_count": 100,
        },
        "error": None,
        "cached": False,
    }


def _factor(
    name: str,
    expression: str,
    rank_ic: float,
    *,
    source: str = "ldm_continuous_discovery",
    index: int = 1,
) -> dict:
    return {
        "candidate": {
            "name": name,
            "expression": expression,
            "reason": f"reason for {name}",
            "source": source,
            "round_id": index,
        },
        "canonical": expression,
        "evaluation_index": index,
        "round_id": index,
        "selection_order_within_round": 1,
        "refill_batch": None,
        "train_metrics": {"rank_ic": rank_ic, "daily_count": 20},
    }


def _write_run(path: Path, factors: list[dict], selected: list[dict]) -> None:
    path.mkdir(parents=True)
    (path / "factor_sequence.json").write_text(
        json.dumps({"factors": factors}), encoding="utf-8"
    )
    (path / "top5_combinations.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_round": 10,
                        "selected_top5_validation_order": selected,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_library_deduplicates_and_never_admits_on_test(tmp_path: Path) -> None:
    run42 = tmp_path / "experiment_seed42"
    run123 = tmp_path / "experiment_seed123"
    alpha158 = _factor("SEED", "$open", 0.09, source="alphabench_alpha158")
    a42 = _factor("A", "$close", 0.05, index=4)
    b42 = _factor("B", "$volume", 0.045, index=5)
    a123 = _factor("A_AGAIN", "$close", 0.03, index=2)
    selected = [
        {
            "candidate": a42["candidate"],
            "validation": _result(0.041, "2021-01-01", "2021-12-31"),
            "test": _result(0.01, "2022-01-01", "2022-12-31"),
        },
        {
            "candidate": b42["candidate"],
            "validation": _result(0.02, "2021-01-01", "2021-12-31"),
            "test": _result(0.09, "2022-01-01", "2022-12-31"),
        },
    ]
    _write_run(run42, [alpha158, a42, b42], selected)
    _write_run(run123, [a123], [])

    payload = build_single_factor_library({42: run42, 123: run123})

    assert payload["summary"] == {
        "unique_train_candidates": 2,
        "validation_admitted": 1,
        "validation_pending": 0,
        "validation_failed": 0,
        "validation_below_threshold": 1,
        "with_retrospective_test_evidence": 2,
    }
    by_name = {row["name"]: row for row in payload["factors"]}
    assert by_name["A"]["admitted"] is True
    assert by_name["A"]["occurrence_count"] == 2
    assert by_name["B"]["test_rank_ic"] == 0.09
    assert by_name["B"]["admitted"] is False


def test_extra_validation_ledger_and_writers(tmp_path: Path) -> None:
    run = tmp_path / "experiment_seed42"
    factor = _factor("A", "$close", 0.05)
    _write_run(run, [factor], [])
    ledger = tmp_path / "single_factor_validation_ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "canonical": "$close",
                "candidate": factor["candidate"],
                "validation": _result(0.05, "2021-01-01", "2021-12-31"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_single_factor_library(
        {42: run}, extra_validation_ledgers=[ledger]
    )
    output = tmp_path / "library"
    write_single_factor_library(output, payload)

    assert payload["summary"]["validation_admitted"] == 1
    assert (output / "single_factor_library.json").is_file()
    assert (output / "single_factor_library.csv").is_file()
    assert json.loads((output / "admitted_factors.json").read_text())["factor_count"] == 1
