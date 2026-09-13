import builtins
import json

import numpy as np
import pandas as pd
import pytest

from alpha_research.reporting.group_nav import compute_group_returns, compound_nav


def test_sorting_ties_missing_and_direction():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2022-01-04", "2022-01-05"]), list("abcde")],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame({"factor": [1, 1, 3, 4, np.inf] * 2,
                          "LABEL": [.01, .03, .05, .07, .99] * 2}, index=index)
    result, skipped = compute_group_returns(panel, "factor", groups=2)
    shuffled, _ = compute_group_returns(panel.sample(frac=1, random_state=42), "factor", groups=2)
    pd.testing.assert_frame_equal(result, shuffled)
    assert skipped == []
    assert result["Group-1"].tolist() == pytest.approx([.02, .02])
    assert result["Group-2"].tolist() == pytest.approx([.06, .06])
    assert result["Long-Short"].tolist() == pytest.approx([.04, .04])
    assert result["valid_pairs"].tolist() == [4, 4]
    nav = compound_nav(result[["Group-1", "Group-2", "Long-Short"]])
    assert nav.iloc[-1].tolist() == pytest.approx([1.02**2, 1.06**2, 1.04**2])


def test_invalid_days_not_silently_zero_filled():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2022-01-04", "2022-01-05"]), list("ab")],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame({"factor": [1, 2, 1, 1], "LABEL": [.1, .2, .1, .2]}, index=index)
    result, skipped = compute_group_returns(panel, "factor", groups=2)
    assert len(result) == 1
    assert skipped == [{"date": "2022-01-05", "valid_pairs": 2}]
    with pytest.raises(ValueError, match="nonfinite"):
        compound_nav(pd.DataFrame({"x": [np.nan]}))
    with pytest.raises(ValueError, match="100%"):
        compound_nav(pd.DataFrame({"x": [-1.1]}))


@pytest.fixture
def saved_nav(tmp_path):
    source = tmp_path / "figures/representative_factor_nav"
    source.mkdir(parents=True)
    dates = pd.to_datetime(["2016-01-04", "2016-01-05", "2021-01-04", "2021-01-05",
                            "2022-01-04", "2022-01-05"])
    daily = pd.DataFrame({"Group-1": [.01] * 6, "Group-2": [.03] * 6,
                          "Long-Short": [.02] * 6, "rank_ic": [.04] * 6,
                          "return_date": dates + pd.Timedelta(days=1)}, index=dates)
    daily.index.name = "datetime"
    daily.to_csv(source / "example_daily.csv", float_format="%.12g")
    splits = {"train": ["2016-01-01", "2020-12-29"],
              "validation": ["2021-01-01", "2021-12-29"],
              "test": ["2022-01-01", "2025-12-26"]}
    factor = {"name": "EXAMPLE", "title": "Saved example"}
    columns = ["Group-1", "Group-2", "Long-Short"]
    record = {"daily_count": len(daily),
              "split_metrics": {split: {"rank_ic": .04} for split in splits},
              "full_terminal_nav": compound_nav(daily[columns]).iloc[-1].to_dict(),
              "test_terminal_nav": compound_nav(daily.loc["2022":, columns]).iloc[-1].to_dict()}
    (source / "metadata.json").write_text(json.dumps({
        "config": {"groups": 2, "splits": splits, "factors": [factor]},
        "factors": {"EXAMPLE": record}, "provider": "/unavailable/original/provider",
    }))
    (source / "example_full_nav.csv").write_text("original saved evidence\n")
    return source


def test_default_replots_saved_data_without_qlib_or_rewriting_evidence(saved_nav, tmp_path, monkeypatch):
    from alpha_research.reporting import group_nav

    monkeypatch.setattr(group_nav, "REPO_ROOT", tmp_path)
    original_import = builtins.__import__

    def reject_qlib(name, *args, **kwargs):
        if name == "qlib" or name.startswith("qlib."):
            raise AssertionError("Saved-data plotting must not load Qlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_qlib)
    evidence = {path.name: path.read_bytes() for path in saved_nav.iterdir()}
    output = tmp_path / "replots"
    assert group_nav.main(["--output-dir", str(output), "--format", "png"]) == 0
    assert sorted(path.name for path in output.iterdir()) == ["example_full_nav.png", "example_test_nav.png"]
    assert (output / "example_full_nav.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert {path.name: path.read_bytes() for path in saved_nav.iterdir()} == evidence


def test_missing_saved_daily_csv_fails_before_rendering_with_recovery_command(saved_nav, tmp_path):
    from alpha_research.reporting.group_nav import main

    (saved_nav / "example_daily.csv").unlink()
    output = tmp_path / "replots"
    with pytest.raises(FileNotFoundError, match=r"Missing saved daily CSV.*--recompute --provider PATH"):
        main(["--source-dir", str(saved_nav), "--output-dir", str(output)])
    assert not output.exists()


def test_saved_daily_values_must_match_preserved_metadata(saved_nav, tmp_path):
    from alpha_research.reporting.group_nav import main

    path = saved_nav / "example_daily.csv"
    daily = pd.read_csv(path)
    daily.loc[0, "Group-2"] = .50
    daily.to_csv(path, index=False)
    with pytest.raises(ValueError, match="NAV differs from metadata"):
        main(["--source-dir", str(saved_nav), "--output-dir", str(tmp_path / "replots")])
