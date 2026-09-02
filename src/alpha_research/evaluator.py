"""Qlib and deterministic mock evaluators for formula alpha research."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .canonical import canonical_form
from .formula import FormulaValidator
from .alphabench_runtime import activate_alphabench
from .io import write_json
from .types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
    ValidationSelection,
    aggregate_daily_metrics,
    finite_or_none,
)


EventSink = Callable[[dict[str, Any]], None]


class FactorEvaluator(Protocol):
    def evaluate(self, expression: str, period: Period) -> EvaluationResult: ...

    def evaluate_many(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> tuple[EvaluationResult, ...]: ...

    def evaluate_pool(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> EvaluationResult: ...


def _daily_series(result: EvaluationResult, key: str) -> dict[str, float]:
    """Map date -> daily metric, skipping non-finite entries."""

    series: dict[str, float] = {}
    for record in result.daily_metrics:
        if not isinstance(record, dict):
            continue
        day = record.get("date")
        value = finite_or_none(record.get(key))
        if day is not None and value is not None:
            series[str(day)] = value
    return series


def _series_correlation(left: dict[str, float], right: dict[str, float]) -> float | None:
    """Pearson correlation over the dates the two series share."""

    shared = left.keys() & right.keys()
    if len(shared) < 30:
        return None
    xs = [left[day] for day in shared]
    ys = [right[day] for day in shared]
    n = len(shared)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(value * value for value in dx)
    var_y = sum(value * value for value in dy)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    covariance = sum(a * b for a, b in zip(dx, dy))
    return covariance / math.sqrt(var_x * var_y)


def select_on_validation(
    *,
    archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
    evaluator: FactorEvaluator,
    validation_period: Period,
    factor_budget: int,
    objective: str,
    event_sink: EventSink | None = None,
    cycle_id: str = "static",
    max_correlation: float | None = None,
    backfill_to_budget: bool = True,
) -> ValidationSelection:
    """Rank the completed search archive using validation-only evidence.

    Candidates are de-duplicated by canonical expression, so the same factor
    written in prefix and infix form counts once. When ``max_correlation`` is
    set, selection walks the ranked list greedily and skips any candidate whose
    daily-IC series correlates above the threshold with an already selected
    factor: the pool is scored as an equal-weight rank combination, so near
    duplicates add cost without adding signal.
    """

    emit = event_sink or (lambda _event: None)
    evaluations: list[tuple[FactorCandidate, EvaluationResult]] = []
    seen: set[str] = set()
    unique_candidates: list[FactorCandidate] = []
    for candidate, _train_result in archive:
        key = canonical_form(candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    validation_results = evaluator.evaluate_many(unique_candidates, validation_period)
    for candidate, result in zip(unique_candidates, validation_results):
        evaluations.append((candidate, result))
        emit({
            "event": "validation_evaluation",
            "cycle_id": cycle_id,
            "candidate": candidate.to_dict(),
            "result": result.to_dict(),
        })
    successful = [item for item in evaluations if item[1].success]
    ranked = sorted(
        successful,
        key=lambda item: item[1].metrics.value(objective),
        reverse=True,
    )
    budget = max(1, int(factor_budget))
    daily_key = "rank_ic" if objective.startswith("rank") else "ic"

    selected_items: list[tuple[FactorCandidate, EvaluationResult]] = []
    selected_series: list[dict[str, float]] = []
    skipped: list[dict[str, Any]] = []
    for candidate, result in ranked:
        if len(selected_items) >= budget:
            break
        if max_correlation is None:
            selected_items.append((candidate, result))
            continue
        series = _daily_series(result, daily_key)
        peak: float | None = None
        for existing in selected_series:
            correlation = _series_correlation(series, existing)
            if correlation is None:
                continue
            if peak is None or abs(correlation) > abs(peak):
                peak = correlation
        if peak is not None and abs(peak) > float(max_correlation):
            skipped.append({
                "expression": candidate.expression,
                "max_abs_correlation": abs(peak),
                daily_key: result.metrics.value(objective),
            })
            continue
        selected_items.append((candidate, result))
        selected_series.append(series)

    if backfill_to_budget and len(selected_items) < budget and skipped:
        # The correlation filter must never shrink the pool below the budget:
        # backfill with the highest-ranked candidates it rejected.
        rejected = {item["expression"] for item in skipped}
        for candidate, result in ranked:
            if len(selected_items) >= budget:
                break
            if candidate.expression in rejected and not any(
                candidate.expression == chosen.expression for chosen, _ in selected_items
            ):
                selected_items.append((candidate, result))

    selected = tuple(candidate for candidate, _result in selected_items)
    if not selected:
        raise RuntimeError("no candidate executed successfully on the validation period")
    emit({
        "event": "validation_selection_summary",
        "cycle_id": cycle_id,
        "unique_candidates": len(unique_candidates),
        "successful": len(successful),
        "selected": len(selected),
        "max_correlation": max_correlation,
        "backfill_to_budget": backfill_to_budget,
        "skipped_for_correlation": skipped,
    })
    return ValidationSelection(selected=selected, evaluations=tuple(evaluations))


class QlibFactorEvaluator:
    """Direct, in-process equivalent of AlphaBench's FFO fast evaluator."""

    def __init__(
        self,
        *,
        provider_uri: str | Path,
        market: str,
        label_expression: str,
        validator: FormulaValidator,
        workers: int = 1,
        min_observations: int = 30,
        quantiles: int = 5,
        label_horizon_days: int = 1,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.provider_uri = Path(provider_uri).expanduser().resolve()
        if not self.provider_uri.is_dir():
            raise FileNotFoundError(f"Qlib provider directory not found: {self.provider_uri}")
        self.market = market
        self.label_expression = label_expression
        self.validator = validator
        self.workers = max(1, int(workers))
        self.min_observations = max(2, int(min_observations))
        self.quantiles = max(2, int(quantiles))
        self.label_horizon_days = max(0, int(label_horizon_days))
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self._initialized = False

    def evaluate(self, expression: str, period: Period) -> EvaluationResult:
        expression = str(expression).strip()
        try:
            self.validator.validate(expression)
        except Exception as exc:
            return self._failure(expression, period, f"formula validation failed: {exc}")
        cached = self._read_cache("single", [expression], period)
        if cached is not None:
            return cached
        try:
            data = self._load_features([expression], period)
            result = self._score_frame(data.iloc[:, 0], data.iloc[:, -1], expression, period)
        except Exception as exc:
            result = self._failure(expression, period, f"Qlib evaluation failed: {exc}")
        self._write_cache("single", [expression], period, result)
        return result

    def evaluate_many(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> tuple[EvaluationResult, ...]:
        """Evaluate many formulas with one Qlib feature load when cache misses."""

        expressions = [factor.expression.strip() for factor in factors]
        results: list[EvaluationResult | None] = [None] * len(expressions)
        missing_indices: list[int] = []
        missing_expressions: list[str] = []
        for index, expression in enumerate(expressions):
            try:
                self.validator.validate(expression)
            except Exception as exc:
                results[index] = self._failure(
                    expression, period, f"formula validation failed: {exc}"
                )
                continue
            cached = self._read_cache("single", [expression], period)
            if cached is not None:
                results[index] = cached
            else:
                missing_indices.append(index)
                missing_expressions.append(expression)
        if missing_expressions:
            try:
                data = self._load_features(missing_expressions, period)
                label = data.iloc[:, -1]
                for column, (index, expression) in enumerate(
                    zip(missing_indices, missing_expressions)
                ):
                    result = self._score_frame(data.iloc[:, column], label, expression, period)
                    results[index] = result
                    self._write_cache("single", [expression], period, result)
            except Exception as exc:
                for index, expression in zip(missing_indices, missing_expressions):
                    result = self._failure(expression, period, f"Qlib batch evaluation failed: {exc}")
                    results[index] = result
                    self._write_cache("single", [expression], period, result)
        return tuple(
            result if result is not None else self._failure(expressions[index], period, "missing result")
            for index, result in enumerate(results)
        )

    def evaluate_pool(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> EvaluationResult:
        expressions = list(dict.fromkeys(item.expression.strip() for item in factors if item.expression.strip()))
        label = "EqualWeightRank(" + ",".join(expressions) + ")"
        if not expressions:
            return self._failure(label, period, "factor pool is empty")
        try:
            for expression in expressions:
                self.validator.validate(expression)
        except Exception as exc:
            return self._failure(label, period, f"formula validation failed: {exc}")
        cached = self._read_cache("pool", expressions, period)
        if cached is not None:
            return cached
        try:
            data = self._load_features(expressions, period)
            feature_frame = data.iloc[:, : len(expressions)]
            datetime_level = self._datetime_level(feature_frame.index)
            ranked = feature_frame.groupby(level=datetime_level, group_keys=False).rank(pct=True)
            signal = ranked.mean(axis=1, skipna=True)
            result = self._score_frame(signal, data.iloc[:, -1], label, period)
        except Exception as exc:
            result = self._failure(label, period, f"Qlib pool evaluation failed: {exc}")
        self._write_cache("pool", expressions, period, result)
        return result

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            import qlib
            from qlib.config import REG_CN
        except ImportError as exc:
            raise RuntimeError(
                "pyqlib is not installed in this interpreter; use the companion Qlib data environment"
            ) from exc
        qlib.init(
            provider_uri=str(self.provider_uri),
            region=REG_CN,
            kernels=self.workers,
            joblib_backend="threading",
        )
        self._initialized = True

    def _load_features(self, expressions: list[str], period: Period):
        self._ensure_initialized()
        from qlib.data import D

        instruments = D.instruments(self.market)
        fields = [*expressions, self.label_expression]
        frame = D.features(
            instruments,
            fields,
            start_time=period.start.isoformat(),
            end_time=period.end.isoformat(),
            freq="day",
        )
        if frame is None or frame.empty:
            raise RuntimeError("Qlib returned no observations")
        return frame.replace([float("inf"), float("-inf")], float("nan"))

    def _score_frame(self, factor, label, expression: str, period: Period) -> EvaluationResult:
        import pandas as pd

        frame = pd.concat([factor.rename("factor"), label.rename("label")], axis=1)
        datetime_level = self._datetime_level(frame.index)
        instrument_level = self._instrument_level(frame.index)
        daily: list[dict[str, Any]] = []
        previous_top: set[str] | None = None
        total_observations = 0
        grouped = list(frame.groupby(level=datetime_level, sort=True))
        if self.label_horizon_days:
            grouped = grouped[: -self.label_horizon_days]
        for timestamp, group in grouped:
            clean = group.dropna()
            if len(clean) < self.min_observations:
                continue
            if clean["factor"].nunique() < 2 or clean["label"].nunique() < 2:
                continue
            ic = clean["factor"].corr(clean["label"], method="pearson")
            rank_ic = clean["factor"].corr(clean["label"], method="spearman")
            bucket = max(1, len(clean) // self.quantiles)
            ordered = clean.sort_values("factor")
            spread = ordered.iloc[-bucket:]["label"].mean() - ordered.iloc[:bucket]["label"].mean()
            top_index = ordered.iloc[-bucket:].index
            if instrument_level is None:
                top = {str(item) for item in top_index}
            else:
                top = {str(item) for item in top_index.get_level_values(instrument_level)}
            turnover = None
            if previous_top:
                turnover = 1.0 - len(previous_top & top) / max(1, len(previous_top))
            previous_top = top
            count = int(len(clean))
            total_observations += count
            daily.append({
                "date": str(getattr(timestamp, "date", lambda: timestamp)()),
                "ic": float(ic) if math.isfinite(float(ic)) else None,
                "rank_ic": float(rank_ic) if math.isfinite(float(rank_ic)) else None,
                "quantile_spread": float(spread) if math.isfinite(float(spread)) else None,
                "turnover": turnover,
                "observation_count": count,
            })
        metrics = aggregate_daily_metrics(daily)
        metrics = EvaluationMetrics(
            **{**metrics.to_dict(), "observation_count": total_observations}
        )
        if not daily or metrics.rank_ic is None:
            return self._failure(expression, period, "insufficient valid cross-sectional observations")
        return EvaluationResult(
            success=True,
            expression=expression,
            period=period,
            metrics=metrics,
            daily_metrics=tuple(daily),
        )

    @staticmethod
    def _datetime_level(index) -> str | int:
        names = list(index.names)
        if "datetime" in names:
            return "datetime"
        return 1 if len(names) > 1 else 0

    @staticmethod
    def _instrument_level(index) -> str | int | None:
        names = list(index.names)
        if "instrument" in names:
            return "instrument"
        return 0 if len(names) > 1 else None

    @staticmethod
    def _failure(expression: str, period: Period, error: str) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            expression=expression,
            period=period,
            error=str(error)[:1500],
        )

    def _cache_path(self, kind: str, expressions: list[str], period: Period) -> Path | None:
        if self.cache_dir is None:
            return None
        raw = json.dumps({
            "kind": kind,
            "provider": str(self.provider_uri),
            "market": self.market,
            "label": self.label_expression,
            "expressions": expressions,
            "period": period.to_dict(),
            "min_observations": self.min_observations,
            "quantiles": self.quantiles,
            "label_horizon_days": self.label_horizon_days,
        }, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(
        self,
        kind: str,
        expressions: list[str],
        period: Period,
    ) -> EvaluationResult | None:
        path = self._cache_path(kind, expressions, period)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return EvaluationResult(
                success=bool(data["success"]),
                expression=str(data["expression"]),
                period=period,
                metrics=EvaluationMetrics(**data.get("metrics", {})),
                error=data.get("error"),
                daily_metrics=tuple(data.get("daily_metrics", [])),
                cached=True,
            )
        except Exception:
            return None

    def _write_cache(
        self,
        kind: str,
        expressions: list[str],
        period: Period,
        result: EvaluationResult,
    ) -> None:
        path = self._cache_path(kind, expressions, period)
        if path is not None:
            write_json(path, result.to_dict())


class AlphaBenchFFOEvaluator:
    """Adapter around AlphaBench's official Factor Formula Operator service."""

    def __init__(
        self,
        *,
        alphabench_root: str | Path,
        base_url: str = "http://127.0.0.1:19777",
        market: str = "csi300",
        label: str = "close_return",
        use_cache: bool = True,
        fast: bool = True,
        topk: int = 50,
        n_drop: int = 5,
        timeout: int = 600,
        forward_n: int = 1,
        max_parallel: int = 1,
        max_attempts: int = 3,
    ) -> None:
        activate_alphabench(alphabench_root)
        try:
            from ffo.client.factor_eval_client import FactorEvalClient
        except ImportError as exc:
            raise RuntimeError(
                "AlphaBench FFO dependencies are unavailable; install the pinned checkout "
                "with `python -m pip install -e external/AlphaBench`"
            ) from exc
        self._client = FactorEvalClient(base_url=base_url, timeout=int(timeout) + 60)
        self.base_url = str(base_url).rstrip("/")
        self.market = market
        self.label = label
        self.use_cache = bool(use_cache)
        self.fast = bool(fast)
        self.topk = int(topk)
        self.n_drop = int(n_drop)
        self.timeout = int(timeout)
        self.forward_n = int(forward_n)
        self.max_parallel = max(1, int(max_parallel))
        self.max_attempts = max(1, int(max_attempts))

    def health_check(self) -> bool:
        return bool(self._client.health_check())

    def evaluate(self, expression: str, period: Period) -> EvaluationResult:
        expression = str(expression).strip()
        if not expression:
            return self._failure(expression, period, "factor expression is empty")
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = self._client.evaluate_factor(
                    expression,
                    market=self.market,
                    start_date=period.start.isoformat(),
                    end_date=period.end.isoformat(),
                    label=self.label,
                    use_cache=self.use_cache,
                    topk=self.topk,
                    n_drop=self.n_drop,
                    timeout=self.timeout,
                    fast=self.fast,
                    forward_n=self.forward_n,
                )
                raw = payload[0] if isinstance(payload, list) and payload else payload
                result = self._from_response(expression, period, raw)
                if result.success:
                    return result
                errors.append(f"attempt {attempt}: {result.error}")
            except Exception as exc:
                errors.append(f"attempt {attempt}: AlphaBench FFO request failed: {exc}")
        return self._failure(
            expression,
            period,
            f"FFO failed after {self.max_attempts} attempts; " + " | ".join(errors),
        )

    def evaluate_many(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> tuple[EvaluationResult, ...]:
        # Keep the control flow identical for seeds, generated factors, validation,
        # and test. FFO's own cache handles duplicate backend work. A single
        # evaluation costs roughly ten seconds, so validating a few hundred
        # candidates serially dominates the run; the FFO service is already
        # multi-worker, so fan the requests out and restore the input order.
        expressions = [factor.expression for factor in factors]
        if not expressions:
            return ()
        if self.max_parallel <= 1 or len(expressions) == 1:
            return tuple(self.evaluate(expression, period) for expression in expressions)
        workers = min(self.max_parallel, len(expressions))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return tuple(pool.map(lambda expression: self.evaluate(expression, period), expressions))

    def evaluate_pool(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> EvaluationResult:
        expressions = list(
            dict.fromkeys(factor.expression.strip() for factor in factors if factor.expression.strip())
        )
        label = "AlphaBenchEqualWeight(" + ",".join(expressions) + ")"
        if not expressions:
            return self._failure(label, period, "factor pool is empty")
        if len(expressions) == 1:
            return self.evaluate(expressions[0], period)
        payload = {
            "expression": expressions,
            "market": self.market,
            "start": period.start.isoformat(),
            "end": period.end.isoformat(),
            "label": self.label,
            "fast": self.fast,
            "topk": self.topk,
            "n_drop": self.n_drop,
            "timeout": self.timeout,
            "forward_n": self.forward_n,
            "use_cache": self.use_cache,
        }
        try:
            raw = self._client._make_request("POST", "/factors/portfolio", json=payload)
            if isinstance(raw, list):
                raw = raw[0] if raw else {}
            if isinstance(raw, dict) and "combined_metrics" in raw:
                if int(raw.get("n_valid_factors", 0)) != len(expressions):
                    return self._failure(
                        label,
                        period,
                        f"FFO pool evaluated only {raw.get('n_valid_factors', 0)}/"
                        f"{len(expressions)} factors successfully",
                    )
                raw = {
                    **raw,
                    "metrics": raw.get("combined_metrics"),
                    "daily_metrics": raw.get("combined_daily_metrics", ()),
                }
            return self._from_response(label, period, raw)
        except Exception as exc:
            return self._failure(label, period, f"AlphaBench FFO pool evaluation failed: {exc}")

    @staticmethod
    def _from_response(
        expression: str,
        period: Period,
        raw: Any,
    ) -> EvaluationResult:
        if not isinstance(raw, dict):
            return AlphaBenchFFOEvaluator._failure(
                expression, period, f"unexpected FFO response: {type(raw).__name__}"
            )
        success = bool(raw.get("success"))
        metrics_raw = raw.get("metrics") or {}
        daily_raw = raw.get("daily_metrics") or raw.get("daily_ic") or ()
        daily = tuple(item for item in daily_raw if isinstance(item, dict))
        validation_error = None

        if success and (not isinstance(daily_raw, (list, tuple)) or not daily):
            validation_error = "FFO marked an empty daily IC/RankIC result as successful"
        elif success and len(daily) != len(daily_raw):
            validation_error = "FFO returned malformed daily observations"

        seen_dates: set[str] = set()
        if success and validation_error is None:
            for row in daily:
                date_text = str(row.get("date") or "")[:10]
                try:
                    parsed_date = date.fromisoformat(date_text)
                    ic = float(row["ic"])
                    rank_ic = float(row["rank_ic"])
                except (KeyError, TypeError, ValueError):
                    validation_error = "FFO returned a malformed daily IC/RankIC observation"
                    break
                if parsed_date < period.start or parsed_date > period.end:
                    validation_error = f"FFO returned an out-of-period date: {date_text}"
                    break
                if not math.isfinite(ic) or not math.isfinite(rank_ic):
                    validation_error = f"FFO returned non-finite daily IC/RankIC on {date_text}"
                    break
                if date_text in seen_dates:
                    validation_error = f"FFO returned duplicate daily observations for {date_text}"
                    break
                seen_dates.add(date_text)

        def number(*keys: str) -> float | None:
            for key in keys:
                try:
                    value = float(metrics_raw[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
            return None

        metrics = EvaluationMetrics(
            ic=number("ic"),
            rank_ic=number("rank_ic"),
            icir=number("icir", "ir"),
            rank_icir=number("rank_icir"),
            quantile_spread=number("quantile_spread"),
            turnover=number("turnover"),
            daily_count=int(metrics_raw.get("daily_count") or len(daily)),
            observation_count=int(metrics_raw.get("observation_count") or 0),
        )
        if success and metrics.rank_ic is None:
            validation_error = validation_error or "FFO returned no finite aggregate RankIC"
        reported_dates = metrics_raw.get("n_dates")
        if success and reported_dates is not None:
            try:
                if int(reported_dates) != len(daily):
                    validation_error = (
                        f"FFO daily coverage mismatch: metrics n_dates={int(reported_dates)}, "
                        f"daily_metrics={len(daily)}"
                    )
            except (TypeError, ValueError):
                validation_error = "FFO returned a non-integer n_dates"
        if validation_error is not None:
            success = False
        return EvaluationResult(
            success=success,
            expression=expression,
            period=period,
            metrics=metrics,
            error=None if success else str(
                validation_error or raw.get("error") or "FFO returned no usable metrics"
            )[:1500],
            daily_metrics=daily,
            cached=bool(raw.get("cached", False)),
        )

    @staticmethod
    def _failure(expression: str, period: Period, error: str) -> EvaluationResult:
        return EvaluationResult(False, expression, period, error=str(error)[:1500])


class MockFactorEvaluator:
    """Stable pseudo-metrics for unit tests; never used by real experiments."""

    def __init__(
        self,
        validator: FormulaValidator | None = None,
        min_observations: int = 30,
    ) -> None:
        self.validator = validator
        self.min_observations = min_observations

    def evaluate(self, expression: str, period: Period) -> EvaluationResult:
        if self.validator is not None:
            try:
                self.validator.validate(expression)
            except Exception as exc:
                return EvaluationResult(False, expression, period, error=str(exc))
        return self._result(expression, period)

    def evaluate_many(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> tuple[EvaluationResult, ...]:
        return tuple(self.evaluate(factor.expression, period) for factor in factors)

    def evaluate_pool(
        self,
        factors: Sequence[FactorCandidate],
        period: Period,
    ) -> EvaluationResult:
        expressions = [item.expression for item in factors]
        expression = "EqualWeightRank(" + ",".join(expressions) + ")"
        if not expressions:
            return EvaluationResult(False, expression, period, error="factor pool is empty")
        return self._result(expression, period)

    def _result(self, expression: str, period: Period) -> EvaluationResult:
        digest = hashlib.sha256(expression.encode("utf-8")).digest()
        base = -0.03 + int.from_bytes(digest[:4], "big") / (2**32 - 1) * 0.13
        period_digest = hashlib.sha256(
            f"{period.start}:{period.end}".encode("utf-8")
        ).digest()
        shift = (int.from_bytes(period_digest[:2], "big") / 65535.0 - 0.5) * 0.02
        daily: list[dict[str, Any]] = []
        for index in range(5):
            noise = ((digest[4 + index] / 255.0) - 0.5) * 0.03
            rank_ic = base + shift + noise
            daily.append({
                "date": f"{period.start.isoformat()}+{index}",
                "ic": rank_ic * 0.9,
                "rank_ic": rank_ic,
                "quantile_spread": rank_ic * 0.01,
                "turnover": None if index == 0 else 0.15 + digest[10 + index] / 2550.0,
                "observation_count": self.min_observations,
            })
        metrics = aggregate_daily_metrics(daily)
        return EvaluationResult(
            True,
            expression,
            period,
            metrics=metrics,
            daily_metrics=tuple(daily),
        )
