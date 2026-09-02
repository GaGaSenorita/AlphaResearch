from alpha_research.evaluator import AlphaBenchFFOEvaluator
from alpha_research.types import Period


PERIOD = Period.from_strings("2021-01-04", "2021-01-05")


def _valid_payload(rank_ic=0.0):
    return {
        "success": True,
        "metrics": {"ic": 0.0, "rank_ic": rank_ic, "n_dates": 2},
        "daily_metrics": [
            {"date": "2021-01-04", "ic": 0.0, "rank_ic": rank_ic},
            {"date": "2021-01-05", "ic": 0.0, "rank_ic": rank_ic},
        ],
        "cached": False,
    }


def test_fake_success_with_empty_daily_metrics_is_rejected():
    raw = {
        "success": True,
        "metrics": {"ic": 0.0, "rank_ic": 0.0, "n_dates": 0},
        "daily_metrics": [],
    }
    result = AlphaBenchFFOEvaluator._from_response("$close", PERIOD, raw)
    assert not result.success
    assert "empty daily" in result.error


def test_real_zero_with_complete_daily_metrics_is_accepted():
    result = AlphaBenchFFOEvaluator._from_response("$close", PERIOD, _valid_payload())
    assert result.success
    assert result.metrics.rank_ic == 0.0
    assert result.metrics.daily_count == 2


def test_duplicate_daily_dates_are_rejected():
    raw = _valid_payload()
    raw["daily_metrics"][1]["date"] = "2021-01-04T00:00:00"
    result = AlphaBenchFFOEvaluator._from_response("$close", PERIOD, raw)
    assert not result.success
    assert "duplicate" in result.error


class _RetryClient:
    def __init__(self):
        self.calls = 0

    def evaluate_factor(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls < 3:
            return [{"success": False, "error": "temporary failure"}]
        return [_valid_payload(0.0123)]


def test_evaluator_retries_until_a_verified_result():
    evaluator = AlphaBenchFFOEvaluator.__new__(AlphaBenchFFOEvaluator)
    evaluator._client = _RetryClient()
    evaluator.market = "csi300"
    evaluator.label = "close_return"
    evaluator.use_cache = False
    evaluator.topk = 50
    evaluator.n_drop = 5
    evaluator.timeout = 10
    evaluator.fast = True
    evaluator.forward_n = 1
    evaluator.max_attempts = 3

    result = evaluator.evaluate("$close", PERIOD)
    assert result.success
    assert result.metrics.rank_ic == 0.0123
    assert evaluator._client.calls == 3
