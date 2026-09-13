"""Profiling is Train-only, including explicit periods and leap-day defaults."""

from types import SimpleNamespace

import pytest

from alpha_research.runner import profile_period
from alpha_research.types import Period


def _args(**overrides):
    values = dict(train_start="2016-01-01", train_end="2020-12-29", profile_start=None, profile_end=None)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("changes", [
    {"profile_start": "2019-01-01"},
    {"profile_end": "2020-12-29"},
])
def test_profile_boundaries_must_be_supplied_together(changes):
    with pytest.raises(ValueError, match="must be supplied together"):
        profile_period(_args(**changes))


@pytest.mark.parametrize("start,end", [
    ("2015-12-31", "2020-12-29"),
    ("2019-01-01", "2021-01-01"),
])
def test_explicit_profile_period_cannot_include_validation_or_pretrain_dates(start, end):
    with pytest.raises(ValueError, match="contained in Train"):
        profile_period(_args(profile_start=start, profile_end=end))


def test_explicit_profile_period_is_preserved_within_train():
    assert profile_period(_args(profile_start="2019-01-01", profile_end="2020-12-29")) == Period.from_strings("2019-01-01", "2020-12-29")


def test_default_profile_period_handles_leap_day():
    assert profile_period(_args(train_end="2020-02-29")) == Period.from_strings("2018-02-28", "2020-02-29")


def test_default_profile_period_is_clipped_to_short_train_window():
    assert profile_period(_args(train_start="2020-01-01")) == Period.from_strings("2020-01-01", "2020-12-29")
