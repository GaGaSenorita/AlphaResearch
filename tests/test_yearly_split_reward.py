"""Annual Stage-2 targets retain signed daily RankIC and equal year grouping."""

from types import SimpleNamespace

import pytest

from alpha_research.methods.split_robust.reward import (
    SplitSettings,
    YearlySplitSettings,
    yearly_split_score,
)


def _result(values_by_year, extra=()):
    records = [
        {"date": f"{year}-01-{index + 1:02d}", "rank_ic": value}
        for year, values in values_by_year.items()
        for index, value in enumerate(values)
    ]
    return SimpleNamespace(
        metrics=SimpleNamespace(rank_ic=0.123),
        daily_metrics=records + list(extra),
    )


def test_reward_is_minimum_year_mean_with_unequal_day_counts():
    result = _result(
        {
            2016: [0.4],
            2017: [-0.4, 0.8],
            2018: [0.1, 0.2, 0.3],
            2019: [0.15] * 10,
            2020: [0.5],
        }
    )

    score = yearly_split_score(result)

    assert score.years == ("2016", "2017", "2018", "2019", "2020")
    assert score.yearly_rankic == pytest.approx((0.4, 0.2, 0.2, 0.15, 0.5))
    assert score.year_days == (1, 2, 3, 10, 1)
    assert score.score == pytest.approx(0.15)
    assert score.worst_rankic == score.score
    assert score.worst_year == "2019"
    assert score.usable_days == 17
    assert score.train_rankic == 0.123
    assert score.rejected is None
    assert score.to_dict()["yearly_rankic"]["2019"] == pytest.approx(0.15)
    assert score.to_dict()["year_days"]["2017"] == 2


@pytest.mark.parametrize("missing_values", [[], [None, float("nan"), float("inf")]])
def test_missing_finite_year_is_rejected(missing_values):
    values = {year: [0.1] for year in range(2016, 2021)}
    values[2018] = missing_values

    score = yearly_split_score(_result(values))

    assert score.score == float("-inf")
    assert score.worst_rankic is None
    assert score.worst_year is None
    assert score.usable_days == 4
    assert score.rejected == "missing finite daily rank_ic for: 2018"
    assert score.to_dict()["score"] is None


@pytest.mark.parametrize("worst", [0.0, -0.2])
def test_zero_and_negative_years_are_preserved_without_sign_flip(worst):
    values = {year: [0.3] for year in range(2016, 2021)}
    values[2017] = [worst]

    score = yearly_split_score(_result(values))

    assert score.score == worst
    assert score.yearly_rankic[1] == worst
    assert score.worst_year == "2017"
    assert score.settings["per_year_sign_flip"] is False


def test_outside_train_and_invalid_records_do_not_change_reward():
    values = {year: [0.2] for year in range(2016, 2021)}
    extra = [
        {"date": "2015-12-31", "rank_ic": -0.9},
        {"date": "2021-01-01", "rank_ic": -0.8},
        {"date": "invalid", "rank_ic": -1.0},
        {"date": "2016-01-02", "rank_ic": float("nan")},
        {"date": "2016-01-03", "rank_ic": None},
        "invalid record",
    ]

    score = yearly_split_score(_result(values, extra))

    assert score.score == 0.2
    assert score.usable_days == 5
    assert score.year_days == (1, 1, 1, 1, 1)


def test_annual_settings_document_shared_contract():
    settings = YearlySplitSettings().to_dict()

    assert settings["objective"] == "worst_rankic"
    assert settings["segmentation"] == "calendar_year"
    assert settings["years"] == [str(year) for year in range(2016, 2021)]
    assert settings["aggregation"] == "minimum_of_yearly_mean_signed_rankic"
    assert settings["factor_evaluations_per_score"] == 1
    assert settings["per_year_sign_flip"] is False
    assert SplitSettings().to_dict() == settings
    assert len(SplitSettings().years) == 5


@pytest.mark.parametrize("start,end", [(2015, 2020), (2016, 2021), (2020, 2016)])
def test_annual_settings_reject_other_train_years(start, end):
    with pytest.raises(ValueError, match="fixed to 2016--2020"):
        YearlySplitSettings(start_year=start, end_year=end)


def test_calendar_year_boundary_is_grouped_by_date_not_record_position():
    result = _result({2016: [0.4], 2018: [0.5], 2019: [0.6], 2020: [0.7]}, [
        {"date": "2016-12-31", "rank_ic": -0.2},
        {"date": "2017-01-01", "rank_ic": -0.3},
        {"date": "2020-12-31", "rank_ic": 0.1},
        {"date": "2021-01-01", "rank_ic": -1.0},
    ])
    result.daily_metrics.reverse()
    score = yearly_split_score(result)
    assert score.yearly_rankic == pytest.approx((0.1, -0.3, 0.5, 0.6, 0.4))
    assert score.year_days == (2, 1, 1, 1, 2)
    assert score.worst_year == "2017"
