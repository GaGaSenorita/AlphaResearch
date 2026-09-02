"""Build a leakage-safe single-factor library from continual LDM runs.

The discovery loop is allowed to use Train.  This module therefore treats a
high Train RankIC as a *candidate* signal only.  A factor is admitted to the
library only when an independently measured Validation RankIC clears the
configured threshold.  Test observations are attached for retrospective
audit, but never change admission status.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from alpha_research.canonical import canonical_form
from alpha_research.io import read_json, write_json


LIBRARY_SCHEMA = "alphaldm.single-factor-library.v1"
SEED_SOURCES = frozenset({"alphabench_alpha158"})
RUN_SEED_PATTERN = re.compile(r"_seed(?P<seed>\d+)$")
METRIC_FIELDS = (
    "rank_ic",
    "ic",
    "rank_icir",
    "icir",
    "quantile_spread",
    "turnover",
    "daily_count",
    "observation_count",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def seed_from_run_name(run_dir: Path) -> int:
    match = RUN_SEED_PATTERN.search(run_dir.name)
    if match is None:
        raise ValueError(f"run directory does not end in _seed<N>: {run_dir}")
    return int(match.group("seed"))


def discover_run_dirs(
    runs_root: Path,
    *,
    seeds: Sequence[int] | None = None,
) -> dict[int, Path]:
    """Find one preserved continual-discovery directory for each seed."""

    requested = None if seeds is None else {int(seed) for seed in seeds}
    found: dict[int, Path] = {}
    for path in sorted(runs_root.iterdir() if runs_root.is_dir() else []):
        if not path.is_dir() or not (path / "factor_sequence.json").is_file():
            continue
        try:
            seed = seed_from_run_name(path)
        except ValueError:
            continue
        if requested is not None and seed not in requested:
            continue
        if seed in found:
            raise ValueError(f"multiple run directories found for seed {seed}")
        found[seed] = path
    if requested is not None:
        missing = sorted(requested - set(found))
        if missing:
            raise FileNotFoundError(f"missing factor_sequence.json for seeds {missing}")
    return found


def _canonical(candidate: Mapping[str, Any], explicit: Any = None) -> str:
    if explicit:
        return str(explicit)
    return canonical_form(str(candidate.get("expression", "")))


def _metrics_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    return {field: metrics.get(field) for field in METRIC_FIELDS}


def _evidence_key(split: str, result: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "split": split,
            "period": result.get("period"),
            "success": bool(result.get("success")),
            "metrics": _metrics_summary(result),
            "error": result.get("error"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _add_evidence(
    evidence: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    canonical: str,
    split: str,
    result: Mapping[str, Any],
    seed: int | None,
    checkpoint_round: int | None,
    source: str,
) -> None:
    if not isinstance(result, Mapping):
        return
    key = _evidence_key(split, result)
    record = evidence[canonical][split].get(key)
    if record is None:
        record = {
            "split": split,
            "period": result.get("period"),
            "success": bool(result.get("success")),
            "metrics": _metrics_summary(result),
            "error": result.get("error"),
            "seeds": set(),
            "checkpoint_rounds": set(),
            "sources": set(),
        }
        evidence[canonical][split][key] = record
    if seed is not None:
        record["seeds"].add(int(seed))
    if checkpoint_round is not None:
        record["checkpoint_rounds"].add(int(checkpoint_round))
    record["sources"].add(source)


def _collect_run_evidence(
    run_dir: Path,
    *,
    seed: int,
    evidence: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    validation_ledger = run_dir / "checkpoint_validation_ledger.jsonl"
    for row in _read_jsonl(validation_ledger):
        candidate = row.get("candidate") or {}
        _add_evidence(
            evidence,
            canonical=_canonical(candidate, row.get("canonical")),
            split="validation",
            result=row.get("validation") or {},
            seed=seed,
            checkpoint_round=row.get("checkpoint_round_first_requested"),
            source=f"{run_dir.name}/checkpoint_validation_ledger.jsonl",
        )

    report_path = run_dir / "top5_combinations.json"
    if not report_path.is_file():
        return
    report = read_json(report_path)
    for checkpoint in report.get("checkpoints", []):
        checkpoint_round = checkpoint.get("checkpoint_round")
        for row in checkpoint.get("selected_top5_validation_order", []):
            candidate = row.get("candidate") or {}
            canonical = _canonical(candidate, row.get("canonical"))
            for split in ("validation", "test"):
                if isinstance(row.get(split), Mapping):
                    _add_evidence(
                        evidence,
                        canonical=canonical,
                        split=split,
                        result=row[split],
                        seed=seed,
                        checkpoint_round=checkpoint_round,
                        source=f"{run_dir.name}/top5_combinations.json",
                    )


def _collect_extra_evidence(
    ledgers: Iterable[Path],
    evidence: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    split: str,
) -> None:
    for ledger in ledgers:
        for row in _read_jsonl(ledger):
            candidate = row.get("candidate") or {}
            result = row.get(split) or {}
            _add_evidence(
                evidence,
                canonical=_canonical(candidate, row.get("canonical")),
                split=split,
                result=result,
                seed=row.get("seed"),
                checkpoint_round=row.get("checkpoint_round"),
                source=ledger.name,
            )


def _finalize_evidence(records: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for record in records.values():
        clean = dict(record)
        clean["seeds"] = sorted(record["seeds"])
        clean["checkpoint_rounds"] = sorted(record["checkpoint_rounds"])
        clean["sources"] = sorted(record["sources"])
        finalized.append(clean)
    def sort_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
        rank_ic = _finite((row.get("metrics") or {}).get("rank_ic"))
        return (
            str((row.get("period") or {}).get("start", "")),
            str((row.get("period") or {}).get("end", "")),
            -rank_ic if rank_ic is not None else float("inf"),
        )

    return sorted(finalized, key=sort_key)


def _best_successful_evidence(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    successful = [
        row
        for row in rows
        if row.get("success")
        and _finite((row.get("metrics") or {}).get("rank_ic")) is not None
    ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda row: float((row.get("metrics") or {})["rank_ic"]),
    )


def build_single_factor_library(
    run_dirs: Mapping[int, Path],
    *,
    train_rank_ic_threshold: float = 0.04,
    validation_rank_ic_threshold: float = 0.04,
    test_rank_ic_audit_threshold: float = 0.04,
    include_seed_factors: bool = False,
    extra_validation_ledgers: Sequence[Path] = (),
    extra_test_ledgers: Sequence[Path] = (),
) -> dict[str, Any]:
    """Aggregate, de-duplicate and classify single factors across runs."""

    occurrences_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for seed, run_dir in sorted(run_dirs.items()):
        sequence = read_json(run_dir / "factor_sequence.json")
        for row in sequence.get("factors", []):
            candidate = row.get("candidate") or {}
            source = str(candidate.get("source", ""))
            if not include_seed_factors and source in SEED_SOURCES:
                continue
            canonical = _canonical(candidate, row.get("canonical"))
            occurrences_by_canonical[canonical].append(
                {
                    "seed": int(seed),
                    "run_name": run_dir.name,
                    "evaluation_index": row.get("evaluation_index"),
                    "round_id": row.get("round_id", candidate.get("round_id")),
                    "selection_order_within_round": row.get(
                        "selection_order_within_round"
                    ),
                    "refill_batch": row.get("refill_batch"),
                    "source": source,
                    "name": candidate.get("name"),
                    "expression": candidate.get("expression"),
                    "reason": candidate.get("reason", ""),
                    "train_metrics": row.get("train_metrics") or {},
                }
            )
        _collect_run_evidence(run_dir, seed=seed, evidence=evidence)

    _collect_extra_evidence(extra_validation_ledgers, evidence, split="validation")
    _collect_extra_evidence(extra_test_ledgers, evidence, split="test")

    factors: list[dict[str, Any]] = []
    for canonical, occurrences in occurrences_by_canonical.items():
        ranked_occurrences = [
            row
            for row in occurrences
            if _finite((row.get("train_metrics") or {}).get("rank_ic")) is not None
        ]
        if not ranked_occurrences:
            continue
        best_train = max(
            ranked_occurrences,
            key=lambda row: float(row["train_metrics"]["rank_ic"]),
        )
        best_train_rank_ic = float(best_train["train_metrics"]["rank_ic"])
        if best_train_rank_ic < float(train_rank_ic_threshold):
            continue

        validation_rows = _finalize_evidence(evidence[canonical]["validation"])
        test_rows = _finalize_evidence(evidence[canonical]["test"])
        best_validation = _best_successful_evidence(validation_rows)
        best_test = _best_successful_evidence(test_rows)
        validation_rank_ic = (
            _finite((best_validation.get("metrics") or {}).get("rank_ic"))
            if best_validation is not None
            else None
        )
        if validation_rank_ic is not None:
            validation_status = (
                "validation_admitted"
                if validation_rank_ic >= float(validation_rank_ic_threshold)
                else "validation_below_threshold"
            )
        elif validation_rows:
            validation_status = "validation_failed"
        else:
            validation_status = "validation_pending"

        ordered_occurrences = sorted(
            occurrences,
            key=lambda row: (
                int(row["seed"]),
                int(row.get("evaluation_index") or 10**12),
            ),
        )
        factor_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        factors.append(
            {
                "factor_id": factor_id,
                "canonical": canonical,
                "name": best_train.get("name"),
                "expression": best_train.get("expression"),
                "reason": best_train.get("reason", ""),
                "source": best_train.get("source"),
                "validation_status": validation_status,
                "admitted": validation_status == "validation_admitted",
                "best_train_rank_ic": best_train_rank_ic,
                "validation_rank_ic": validation_rank_ic,
                "test_rank_ic": (
                    _finite((best_test.get("metrics") or {}).get("rank_ic"))
                    if best_test is not None
                    else None
                ),
                "best_train_occurrence": best_train,
                "occurrence_count": len(ordered_occurrences),
                "occurrences": ordered_occurrences,
                "validation_evidence": validation_rows,
                "test_evidence": test_rows,
            }
        )

    status_priority = {
        "validation_admitted": 0,
        "validation_pending": 1,
        "validation_failed": 2,
        "validation_below_threshold": 3,
    }
    factors.sort(
        key=lambda row: (
            status_priority[row["validation_status"]],
            -(row["validation_rank_ic"] if row["validation_rank_ic"] is not None else -1e9),
            -row["best_train_rank_ic"],
            str(row["name"]),
        )
    )
    counts = {
        status: sum(row["validation_status"] == status for row in factors)
        for status in status_priority
    }
    return {
        "schema_version": LIBRARY_SCHEMA,
        "policy": {
            "candidate_rule": f"Train RankIC >= {float(train_rank_ic_threshold):.6g}",
            "train_rank_ic_threshold": float(train_rank_ic_threshold),
            "admission_rule": (
                f"Validation RankIC >= {float(validation_rank_ic_threshold):.6g}"
            ),
            "validation_rank_ic_threshold": float(validation_rank_ic_threshold),
            "test_usage": "retrospective audit only; never used for admission",
            "test_audit_threshold": float(test_rank_ic_audit_threshold),
            "include_seed_factors": bool(include_seed_factors),
            "deduplication": "canonical expression across all selected seeds",
        },
        "source_runs": [
            {"seed": int(seed), "run_name": run_dir.name}
            for seed, run_dir in sorted(run_dirs.items())
        ],
        "summary": {
            "unique_train_candidates": len(factors),
            "validation_admitted": counts["validation_admitted"],
            "validation_pending": counts["validation_pending"],
            "validation_failed": counts["validation_failed"],
            "validation_below_threshold": counts["validation_below_threshold"],
            "with_retrospective_test_evidence": sum(
                row["test_rank_ic"] is not None for row in factors
            ),
            "pending_retrospective_test": sum(
                row["test_rank_ic"] is None for row in factors
            ),
            "retrospective_test_at_least_threshold": sum(
                row["test_rank_ic"] is not None
                and row["test_rank_ic"] >= float(test_rank_ic_audit_threshold)
                for row in factors
            ),
            "train_validation_test_at_least_threshold": sum(
                row["validation_rank_ic"] is not None
                and row["validation_rank_ic"] >= float(validation_rank_ic_threshold)
                and row["test_rank_ic"] is not None
                and row["test_rank_ic"] >= float(test_rank_ic_audit_threshold)
                for row in factors
            ),
        },
        "factors": factors,
    }


def _csv_rows(factors: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for factor in factors:
        best = factor.get("best_train_occurrence") or {}
        yield {
            "factor_id": factor.get("factor_id"),
            "name": factor.get("name"),
            "expression": factor.get("expression"),
            "reason": factor.get("reason"),
            "validation_status": factor.get("validation_status"),
            "admitted": factor.get("admitted"),
            "train_rank_ic": factor.get("best_train_rank_ic"),
            "validation_rank_ic": factor.get("validation_rank_ic"),
            "test_rank_ic_retrospective": factor.get("test_rank_ic"),
            "best_seed": best.get("seed"),
            "best_round": best.get("round_id"),
            "best_evaluation_index": best.get("evaluation_index"),
            "occurrence_count": factor.get("occurrence_count"),
            "all_seeds": json.dumps(
                sorted({row["seed"] for row in factor.get("occurrences", [])})
            ),
            "canonical": factor.get("canonical"),
        }


def _write_csv(path: Path, factors: Sequence[Mapping[str, Any]]) -> None:
    rows = list(_csv_rows(factors))
    fieldnames = list(rows[0]) if rows else [
        "factor_id",
        "name",
        "expression",
        "reason",
        "validation_status",
        "admitted",
        "train_rank_ic",
        "validation_rank_ic",
        "test_rank_ic_retrospective",
        "best_seed",
        "best_round",
        "best_evaluation_index",
        "occurrence_count",
        "all_seeds",
        "canonical",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_single_factor_library(output_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write the complete candidate table and the admitted-only view."""

    output_dir.mkdir(parents=True, exist_ok=True)
    factors = list(payload.get("factors", []))
    admitted = [row for row in factors if row.get("admitted")]
    test_threshold = float((payload.get("policy") or {}).get("test_audit_threshold", 0.04))
    test_audit_over_threshold = [
        row
        for row in factors
        if row.get("test_rank_ic") is not None
        and float(row["test_rank_ic"]) >= test_threshold
    ]
    three_split_audit = [
        row
        for row in test_audit_over_threshold
        if row.get("validation_rank_ic") is not None
        and float(row["validation_rank_ic"])
        >= float(
            (payload.get("policy") or {}).get("validation_rank_ic_threshold", 0.04)
        )
    ]
    write_json(output_dir / "single_factor_library.json", payload)
    _write_csv(output_dir / "single_factor_library.csv", factors)
    admitted_payload = {
        "schema_version": LIBRARY_SCHEMA,
        "policy": payload.get("policy"),
        "source_runs": payload.get("source_runs"),
        "factor_count": len(admitted),
        "factors": admitted,
    }
    write_json(output_dir / "admitted_factors.json", admitted_payload)
    _write_csv(output_dir / "admitted_factors.csv", admitted)
    test_audit_payload = {
        "schema_version": LIBRARY_SCHEMA,
        "warning": "retrospective Test audit only; never used for factor admission",
        "threshold": test_threshold,
        "factor_count": len(test_audit_over_threshold),
        "factors": test_audit_over_threshold,
    }
    write_json(output_dir / "test_audit_over_threshold.json", test_audit_payload)
    _write_csv(
        output_dir / "test_audit_over_threshold.csv", test_audit_over_threshold
    )
    three_split_payload = {
        "schema_version": LIBRARY_SCHEMA,
        "warning": (
            "post-hoc three-split audit; Test was not used during discovery or "
            "formal Validation admission"
        ),
        "factor_count": len(three_split_audit),
        "factors": three_split_audit,
    }
    write_json(output_dir / "three_split_audit_over_threshold.json", three_split_payload)
    _write_csv(output_dir / "three_split_audit_over_threshold.csv", three_split_audit)

    summary = payload.get("summary") or {}
    readme = f"""# AlphaLDM single-factor library

This directory is derived from the ordered `factor_sequence.json` artifacts.

- Candidate gate: `{payload['policy']['candidate_rule']}`
- Admission gate: `{payload['policy']['admission_rule']}`
- Test: {payload['policy']['test_usage']}
- Unique Train candidates: {summary.get('unique_train_candidates', 0)}
- Validation-admitted factors: {summary.get('validation_admitted', 0)}
- Pending Validation: {summary.get('validation_pending', 0)}
- Below the Validation threshold: {summary.get('validation_below_threshold', 0)}
- Measured on Test: {summary.get('with_retrospective_test_evidence', 0)}
- Test audit >= {test_threshold:.6g}: {summary.get('retrospective_test_at_least_threshold', 0)}
- Train/Validation/Test all above threshold: {summary.get('train_validation_test_at_least_threshold', 0)}

`single_factor_library.*` contains every Train candidate and its status.
`admitted_factors.*` is the strict factor-library view.  Formula, explanation,
seed, round and original evaluation order are retained.  Test values, when
present, are audit metadata and cannot admit a factor.
`test_audit_over_threshold.*` is a descriptive retrospective view, not a
selection or admission result.
`three_split_audit_over_threshold.*` contains the post-hoc intersection of the
Train, Validation and Test thresholds and carries the same Test-audit warning.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
