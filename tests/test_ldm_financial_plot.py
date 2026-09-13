from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


from alpha_research.reporting import data as PLOT

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_renderer_preserves_measurements_and_saves_one_pdf(tmp_path, seed):
    from alpha_research.reporting.financial import draw_financial_trajectory

    run = ROOT / "runs/ldm_continuous_discovery" / (
        f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
    )
    old_prefix = ROOT / "figures" / f"ldm_financial_mining_seed{seed}_round100"
    actual = draw_financial_trajectory(run, tmp_path / "figure", seed=seed)
    expected = PLOT.load_json(old_prefix.with_suffix(".json"))
    # Source paths relocate with a source export; every plotted measurement and
    # interpretation flag must still match the preserved report.
    location_fields = {"outputs", "data_dir"}
    assert {k: v for k, v in actual.items() if k not in location_fields} == {
        k: v for k, v in expected.items() if k not in location_fields
    }
    assert actual["data_dir"] == str(run.resolve())
    assert set(actual["outputs"]) == {"pdf", "metadata"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["figure.json", "figure.pdf"]
    assert (tmp_path / "figure.pdf").read_bytes().startswith(b"%PDF-")


def test_retrospective_test_best_so_far_keeps_raw_values_and_source_rounds():
    records = PLOT.retrospective_test_best_so_far(
        [0, 10, 20, 30],
        [0.030, 0.040, 0.035, 0.041],
    )
    assert [row["rank_ic"] for row in records] == [0.030, 0.040, 0.040, 0.041]
    assert [row["raw_rank_ic"] for row in records] == [0.030, 0.040, 0.035, 0.041]
    assert [row["source_round"] for row in records] == [0, 10, 10, 30]
    assert [row["improved"] for row in records] == [True, True, False, True]


def test_regular_checkpoint_rows_hides_legacy_off_grid_audits():
    rows = [
        {"checkpoint_round": round_id}
        for round_id in [20, 38, 50, 75, 100, 0, 10, 30, 40, 60, 70, 80, 90]
    ]
    selected = PLOT.regular_checkpoint_rows(rows, snapshot_round=100, step=10)
    assert [row["checkpoint_round"] for row in selected] == list(range(0, 101, 10))


def test_five_round_grid_keeps_r75_but_not_r38():
    rows = [{"checkpoint_round": r} for r in [38, *reversed(range(0, 101, 5))]]
    selected = PLOT.regular_checkpoint_rows(rows, 100, require_complete=True)
    assert [row["checkpoint_round"] for row in selected] == list(range(0, 101, 5))


def test_five_round_plot_rejects_incomplete_or_duplicate_reports():
    with pytest.raises(ValueError, match="missing measured checkpoints"):
        PLOT.regular_checkpoint_rows(
            [{"checkpoint_round": r} for r in range(0, 101, 10)],
            100, require_complete=True,
        )
    with pytest.raises(ValueError, match="duplicate"):
        PLOT.regular_checkpoint_rows([{"checkpoint_round": 0}] * 2, 0)


@pytest.mark.parametrize("success,value", [(False, 0.0), (True, None), (True, float("nan"))])
def test_plot_rejects_invalid_test_measurement(success, value):
    with pytest.raises(ValueError, match="invalid measured Test"):
        PLOT.measured_test_values([{
            "checkpoint_round": 5,
            "test_equal_weight_rank_pool": {"success": success, "metrics": {"rank_ic": value}},
        }])


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_preserved_five_round_reports_are_complete_and_do_not_rewrite_search(seed):
    root = ROOT
    run = root / "runs/ldm_continuous_discovery" / (
        f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
    )
    imported = PLOT.load_json(run / "five_round_checkpoint_import.json")
    for name in ("resume_state.json", "factor_sequence.json"):
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == imported["source_files_sha256"][name]
    report = PLOT.load_json(run / "top5_combinations.json")
    by_round = {row["checkpoint_round"]: row for row in report["checkpoints"]}
    for r, digest in imported["original_checkpoint_sha256"].items():
        assert hashlib.sha256(json.dumps(by_round[int(r)], sort_keys=True).encode()).hexdigest() == digest
    checkpoints = PLOT.regular_checkpoint_rows(report["checkpoints"], 100, require_complete=True)
    assert len(checkpoints) == 21
    rounds = list(range(0, 101, 5))
    factors = PLOT.load_json(run / "factor_sequence.json")["factors"]
    for row in checkpoints:
        r = row["checkpoint_round"]
        assert row["available_factor_count"] == sum(f["round_id"] <= r for f in factors)
        assert len(row["selected_top5_validation_order"]) == 5
        assert all(f["candidate"]["round_id"] <= r for f in row["selected_top5_validation_order"])
    raw = PLOT.measured_test_values(checkpoints)
    expected = PLOT.retrospective_test_best_so_far(rounds, raw)
    plotted = PLOT.load_json(root / "figures" / f"ldm_financial_mining_seed{seed}_round100.json")
    assert plotted["checkpoint_step"] == 5
    assert [p["round"] for p in plotted["train_milestones"]] == rounds
    assert plotted["train_milestones"] == PLOT.train_leaders_at_rounds(factors, rounds)
    assert [p["round"] for p in plotted["measured_top5_test_checkpoints"]] == rounds
    assert [p["rank_ic"] for p in plotted["measured_top5_test_checkpoints"]] == raw
    assert plotted["retrospective_test_best_so_far"] == expected
    assert plotted["single_factor_points_displayed"] is False
    assert plotted["illustrative_projection"] is None
    combined = PLOT.load_json(root / "figures/ldm_financial_mining_three_seed_comparison.json")
    entry = next(row for row in combined["seeds"] if row["seed"] == seed)
    assert entry["train"] == plotted["train_milestones"]
    assert entry["raw_test"] == plotted["measured_top5_test_checkpoints"]
    assert entry["test_best_so_far"] == expected
    assert set(plotted["outputs"]) == {"pdf", "metadata"}
    assert set(combined["outputs"]) == {"pdf", "metadata"}
    assert (root / "figures" / f"ldm_financial_mining_seed{seed}_round100.pdf").is_file()
