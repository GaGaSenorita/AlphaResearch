"""Deterministic diversity diagnostics for the HarnessLDM candidate archive.

The controller is deliberately separate from AlphaLDM's numeric ``History``:
it never changes a Train score, GP target, posterior, or acquisition value.
Structural rejection happens before FFO.  Train-only behavioural information
is available only after FFO and can therefore influence the next prompt, not
the evaluation that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from alpha_research.canonical import canonical_form
from alpha_research.formula import ROLLING_ONE, ROLLING_TWO, FormulaValidator, Node


DIVERSITY_MODES = ("off", "diagnostics", "feedback", "reject", "penalty")
DIVERSITY_SCHEMA_VERSION = "alphaldm.diversity.v1"
_COMMUTATIVE = {"Add", "Mul", "And", "Or", "Eq", "Ne", "Greater", "Less"}


def _validate_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in DIVERSITY_MODES:
        raise ValueError(f"diversity mode must be one of {DIVERSITY_MODES}, got {value!r}")
    return mode


@dataclass(frozen=True)
class DiversitySettings:
    structural_mode: str = "off"
    behavioral_mode: str = "off"
    structural_similarity_threshold: float | None = None
    behavioral_correlation_threshold: float | None = None
    penalty_weight: float = 0.0
    feedback_clusters: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_mode", _validate_mode(self.structural_mode))
        object.__setattr__(self, "behavioral_mode", _validate_mode(self.behavioral_mode))
        for name in ("structural_similarity_threshold", "behavioral_correlation_threshold"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if float(self.penalty_weight) < 0:
            raise ValueError("penalty_weight must be non-negative")
        if int(self.feedback_clusters) < 1:
            raise ValueError("feedback_clusters must be positive")

    @property
    def active(self) -> bool:
        return self.structural_mode != "off" or self.behavioral_mode != "off"


def _number_text(value: str) -> str:
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


def _is_window_argument(parent: Node | None, index: int | None) -> bool:
    if parent is None or parent.kind != "call" or index is None:
        return False
    if parent.value in ROLLING_ONE:
        return index == 1
    if parent.value in ROLLING_TWO:
        return index == 2
    if parent.value == "Quantile":
        return index == 1
    return False


def _render(
    node: Node,
    *,
    normalize_parameters: bool,
    parent: Node | None = None,
    index: int | None = None,
) -> str:
    if node.kind == "field":
        return node.value
    if node.kind == "number":
        if normalize_parameters:
            return "<WINDOW>" if _is_window_argument(parent, index) else "<PARAM>"
        return _number_text(node.value)
    rendered = [
        _render(
            child,
            normalize_parameters=normalize_parameters,
            parent=node,
            index=child_index,
        )
        for child_index, child in enumerate(node.args)
    ]
    if node.value in _COMMUTATIVE and len(rendered) == 2:
        rendered.sort()
    return f"{node.value}({','.join(rendered)})"


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.args:
        yield from _walk(child)


def _depth(node: Node) -> int:
    return 1 if not node.args else 1 + max(_depth(child) for child in node.args)


def _subtree_table(root: Node) -> dict[str, int]:
    table: dict[str, int] = {}
    for node in _walk(root):
        signature = _render(node, normalize_parameters=True)
        size = sum(1 for _ in _walk(node))
        table[signature] = max(size, table.get(signature, 0))
    return table


def _short_cluster_id(prefix: str, signature: str) -> str:
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class StructuralProfile:
    canonical_ast: str
    parameter_normalized_ast: str
    subtree_signature: str
    node_count: int
    depth: int
    operators: tuple[str, ...]
    operator_counts: dict[str, int]
    fields: tuple[str, ...]
    window_count: int
    parameter_count: int
    subtree_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operators"] = list(self.operators)
        data["fields"] = list(self.fields)
        return data


class StructuralAnalyzer:
    """Reuse ``FormulaValidator``'s strict AST; do not maintain another parser."""

    def __init__(self, validator: FormulaValidator) -> None:
        self.validator = validator

    def profile(self, expression: str) -> StructuralProfile:
        canonical = canonical_form(expression)
        root = self.validator.validate(canonical)
        nodes = list(_walk(root))
        calls = [node.value for node in nodes if node.kind == "call"]
        fields = sorted({node.value for node in nodes if node.kind == "field"})
        parameters = [node for node in nodes if node.kind == "number"]
        windows = 0
        for node in nodes:
            if node.kind != "call":
                continue
            windows += sum(
                1
                for index, child in enumerate(node.args)
                if child.kind == "number" and _is_window_argument(node, index)
            )
        subtrees = _subtree_table(root)
        root_signature = _render(root, normalize_parameters=True)
        proper = [(size, signature) for signature, size in subtrees.items() if signature != root_signature]
        core = max(proper, default=(subtrees[root_signature], root_signature), key=lambda item: (item[0], item[1]))[1]
        return StructuralProfile(
            canonical_ast=canonical,
            parameter_normalized_ast=root_signature,
            subtree_signature=core,
            node_count=len(nodes),
            depth=_depth(root),
            operators=tuple(sorted(set(calls))),
            operator_counts=dict(sorted(Counter(calls).items())),
            fields=tuple(fields),
            window_count=windows,
            parameter_count=len(parameters),
            subtree_count=len(subtrees),
        )

    def similarity(self, left: StructuralProfile, right: StructuralProfile) -> tuple[float, int, str | None]:
        left_root = self.validator.validate(left.canonical_ast)
        right_root = self.validator.validate(right.canonical_ast)
        left_subtrees = _subtree_table(left_root)
        right_subtrees = _subtree_table(right_root)
        common = set(left_subtrees).intersection(right_subtrees)
        if not common:
            return 0.0, 0, None
        signature = max(common, key=lambda item: (min(left_subtrees[item], right_subtrees[item]), item))
        common_nodes = min(left_subtrees[signature], right_subtrees[signature])
        denominator = max(1, min(left.node_count, right.node_count))
        return float(common_nodes / denominator), int(common_nodes), signature


def train_rank_ic_series(result: Any) -> dict[str, float]:
    """Extract only the Train result supplied by the search loop."""
    series: dict[str, float] = {}
    for row in getattr(result, "daily_metrics", ()) or ():
        try:
            value = float(row.get("rank_ic"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and row.get("date") is not None:
            series[str(row["date"])] = value
    return series


def series_correlation(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    dates = sorted(set(left).intersection(right))
    if len(dates) < 3:
        return None
    x = np.asarray([left[date] for date in dates], dtype=float)
    y = np.asarray([right[date] for date in dates], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x, y = x[finite], y[finite]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass
class ArchiveEntry:
    key: str
    expression: str
    name: str
    source: str
    round_id: int
    train_score: float | None
    structural: StructuralProfile
    train_rank_ic: dict[str, float]
    nearest_structural_neighbor: str | None = None
    structural_similarity: float | None = None
    largest_common_subtree_nodes: int | None = None
    nearest_behavioral_neighbor: str | None = None
    behavioral_correlation: float | None = None
    max_abs_behavioral_correlation: float | None = None
    behavioral_prompt_excluded: bool = False


class DiversityController:
    """Structural and Train-only behavioural archive for one HarnessLDM run."""

    def __init__(
        self,
        *,
        validator: FormulaValidator,
        settings: DiversitySettings,
        output_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.analyzer = StructuralAnalyzer(validator)
        self.entries: dict[str, ArchiveEntry] = {}
        self._proposal_profiles: dict[str, StructuralProfile] = {}
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.events_path = self.output_dir / "diversity_diagnostics.jsonl" if self.output_dir else None
        if self.events_path is not None and self.events_path.exists():
            raise FileExistsError(
                f"refusing to append to existing diversity diagnostics: {self.events_path}"
            )

    @property
    def active(self) -> bool:
        return self.settings.active

    def _emit(self, event: dict[str, Any]) -> None:
        if self.events_path is None:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _nearest_structural(
        self,
        profile: StructuralProfile,
        extra: Sequence[tuple[str, StructuralProfile]] = (),
    ) -> tuple[str | None, float, int, str | None]:
        nearest: tuple[str | None, float, int, str | None] = (None, 0.0, 0, None)
        candidates = [(entry.expression, entry.structural) for entry in self.entries.values()]
        candidates.extend(extra)
        for expression, other in candidates:
            similarity, nodes, subtree = self.analyzer.similarity(profile, other)
            if (similarity, nodes, expression) > (nearest[1], nearest[2], nearest[0] or ""):
                nearest = (expression, similarity, nodes, subtree)
        return nearest

    def assess_proposal(
        self,
        candidate: Mapping[str, Any],
        batch_profiles: Sequence[tuple[str, StructuralProfile]] = (),
    ) -> tuple[StructuralProfile, bool]:
        expression = str(candidate.get("expression", ""))
        profile = self.analyzer.profile(expression)
        nearest, similarity, common_nodes, common_subtree = self._nearest_structural(profile, batch_profiles)
        existing_profiles = [entry.structural for entry in self.entries.values()] + [item[1] for item in batch_profiles]
        exact_duplicate = any(profile.canonical_ast == item.canonical_ast for item in existing_profiles)
        parameter_variant = any(
            profile.parameter_normalized_ast == item.parameter_normalized_ast
            and profile.canonical_ast != item.canonical_ast
            for item in existing_profiles
        )
        threshold_hit = (
            self.settings.structural_similarity_threshold is not None
            and similarity >= float(self.settings.structural_similarity_threshold)
        )
        rejected = self.settings.structural_mode == "reject" and (
            exact_duplicate or parameter_variant or threshold_hit
        )
        self._proposal_profiles[profile.canonical_ast] = profile
        self._emit({
            "event": "structural_diversity_proposal",
            "candidate": dict(candidate),
            "structural": profile.to_dict(),
            "nearest_structural_neighbor": nearest,
            "structural_similarity": similarity,
            "largest_common_subtree_nodes": common_nodes,
            "largest_common_subtree": common_subtree,
            "exact_duplicate": exact_duplicate,
            "parameter_only_variant": parameter_variant,
            "mode": self.settings.structural_mode,
            "rejected": rejected,
        })
        return profile, rejected

    def record_train_evaluation(
        self,
        *,
        candidate: Mapping[str, Any],
        result: Any,
        score: float | None,
    ) -> dict[str, Any] | None:
        if not bool(getattr(result, "success", False)):
            return None
        expression = str(candidate.get("expression", ""))
        profile = self._proposal_profiles.get(canonical_form(expression)) or self.analyzer.profile(expression)
        key = profile.canonical_ast
        if key in self.entries:
            return None
        nearest, similarity, common_nodes, _subtree = self._nearest_structural(profile)
        behavioral_active = self.settings.behavioral_mode != "off"
        series = train_rank_ic_series(result) if behavioral_active else {}
        behavioral_neighbor: str | None = None
        signed_correlation: float | None = None
        max_abs = -1.0
        for entry in self.entries.values() if behavioral_active else ():
            correlation = series_correlation(series, entry.train_rank_ic)
            if correlation is not None and abs(correlation) > max_abs:
                max_abs = abs(correlation)
                signed_correlation = correlation
                behavioral_neighbor = entry.expression
        if max_abs < 0:
            max_abs_value: float | None = None
        else:
            max_abs_value = float(max_abs)
        behavioral_threshold_hit = (
            self.settings.behavioral_correlation_threshold is not None
            and max_abs_value is not None
            and max_abs_value >= float(self.settings.behavioral_correlation_threshold)
        )
        excluded = self.settings.behavioral_mode == "reject" and behavioral_threshold_hit
        entry = ArchiveEntry(
            key=key,
            expression=expression,
            name=str(candidate.get("name", "")),
            source=str(candidate.get("source", "")),
            round_id=int(candidate.get("round_id", 0) or 0),
            train_score=float(score) if score is not None and math.isfinite(float(score)) else None,
            structural=profile,
            train_rank_ic=series,
            nearest_structural_neighbor=nearest,
            structural_similarity=similarity if nearest is not None else None,
            largest_common_subtree_nodes=common_nodes if nearest is not None else None,
            nearest_behavioral_neighbor=behavioral_neighbor,
            behavioral_correlation=signed_correlation,
            max_abs_behavioral_correlation=max_abs_value,
            behavioral_prompt_excluded=excluded,
        )
        self.entries[key] = entry
        clusters = self.cluster_labels()
        event = {
            "event": "train_only_diversity_evaluation",
            "candidate": dict(candidate),
            "train_period_only": True,
            "structural": profile.to_dict(),
            "nearest_structural_neighbor": nearest,
            "structural_similarity": entry.structural_similarity,
            "largest_common_subtree_nodes": entry.largest_common_subtree_nodes,
            "structural_cluster": clusters["structural"].get(key),
            "nearest_behavioral_neighbor": behavioral_neighbor,
            "behavioral_correlation": signed_correlation,
            "max_abs_behavioral_correlation": max_abs_value,
            "behavioral_cluster": clusters["behavioral"].get(key),
            "behavioral_prompt_excluded": excluded,
            "structural_mode": self.settings.structural_mode,
            "behavioral_mode": self.settings.behavioral_mode,
        }
        self._emit(event)
        return event

    def cluster_labels(
        self,
        *,
        structural_threshold: float | None = None,
        behavioral_threshold: float | None = None,
    ) -> dict[str, dict[str, str]]:
        keys = sorted(self.entries)
        structural_uf = _UnionFind(keys)
        behavioral_uf = _UnionFind(keys)
        st = self.settings.structural_similarity_threshold if structural_threshold is None else structural_threshold
        bt = self.settings.behavioral_correlation_threshold if behavioral_threshold is None else behavioral_threshold
        for left_index, left_key in enumerate(keys):
            left = self.entries[left_key]
            for right_key in keys[left_index + 1 :]:
                right = self.entries[right_key]
                if left.structural.parameter_normalized_ast == right.structural.parameter_normalized_ast:
                    structural_uf.union(left_key, right_key)
                elif st is not None and self.analyzer.similarity(left.structural, right.structural)[0] >= float(st):
                    structural_uf.union(left_key, right_key)
                if bt is not None:
                    corr = series_correlation(left.train_rank_ic, right.train_rank_ic)
                    if corr is not None and abs(corr) >= float(bt):
                        behavioral_uf.union(left_key, right_key)
        structural = {
            key: _short_cluster_id("S", structural_uf.find(key))
            for key in keys
        }
        behavioral = {
            key: _short_cluster_id("B", behavioral_uf.find(key))
            for key in keys
        }
        return {"structural": structural, "behavioral": behavioral}

    def include_in_prompt_examples(self, expression: str) -> bool:
        entry = self.entries.get(canonical_form(expression))
        return entry is None or not entry.behavioral_prompt_excluded

    def prompt_adjusted_score(self, expression: str, score: float) -> float:
        """Prompt-example priority only; never returned to AlphaLDM History."""
        if self.settings.penalty_weight <= 0:
            return float(score)
        entry = self.entries.get(canonical_form(expression))
        if entry is None:
            return float(score)
        penalty = 0.0
        if self.settings.structural_mode == "penalty" and entry.structural_similarity is not None:
            penalty += float(entry.structural_similarity)
        if self.settings.behavioral_mode == "penalty" and entry.max_abs_behavioral_correlation is not None:
            penalty += float(entry.max_abs_behavioral_correlation)
        return float(score) - float(self.settings.penalty_weight) * penalty

    @staticmethod
    def _groups(labels: Mapping[str, str]) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, label in labels.items():
            grouped[label].append(key)
        return sorted(grouped.values(), key=lambda group: (-len(group), min(group)))

    def compressed_feedback(self) -> dict[str, Any]:
        labels = self.cluster_labels()
        limit = int(self.settings.feedback_clusters)
        structural_families = []
        if self.settings.structural_mode in {"feedback", "reject", "penalty"}:
            for group in self._groups(labels["structural"]):
                if len(group) < 2:
                    continue
                entries = [self.entries[key] for key in group]
                representative = max(entries, key=lambda item: item.train_score if item.train_score is not None else -math.inf)
                structural_families.append({
                    "cluster": labels["structural"][group[0]],
                    "count": len(group),
                    "representative_expression": representative.expression,
                    "normalized_signature": representative.structural.parameter_normalized_ast,
                    "operators": list(representative.structural.operators),
                    "fields": list(representative.structural.fields),
                    "instruction": "Use a different core operator/field construction, not another parameter variant.",
                })
        behavioral_families = []
        if (
            self.settings.behavioral_mode in {"feedback", "reject", "penalty"}
            and self.settings.behavioral_correlation_threshold is not None
        ):
            for group in self._groups(labels["behavioral"]):
                if len(group) < 2:
                    continue
                entries = [self.entries[key] for key in group]
                representative = max(entries, key=lambda item: item.train_score if item.train_score is not None else -math.inf)
                correlations = [
                    entry.max_abs_behavioral_correlation
                    for entry in entries
                    if entry.max_abs_behavioral_correlation is not None
                ]
                behavioral_families.append({
                    "cluster": labels["behavioral"][group[0]],
                    "count": len(group),
                    "representative_expression": representative.expression,
                    "max_abs_train_rank_ic_correlation": max(correlations, default=None),
                    "instruction": "Propose a different Train-signal family, not a cosmetic structural rewrite.",
                })
        return {
            "crowded_structural_families": structural_families[:limit],
            "crowded_train_only_behavioral_families": behavioral_families[:limit],
            "behavioral_correlation_basis": "daily RankIC series on Training only",
        }

    def prompt_feedback_enabled(self) -> bool:
        prompt_modes = {"feedback", "reject", "penalty"}
        return (
            self.settings.structural_mode in prompt_modes
            or self.settings.behavioral_mode in prompt_modes
        )
