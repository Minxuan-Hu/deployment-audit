from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExportSchema:
    name: str
    columns: tuple[str, ...]


AUDIT_CARD_SCHEMA = ExportSchema(
    name="audit_card",
    columns=(
        "benchmark_name",
        "family_name",
        "score_name",
        "n_candidates",
        "n_feasible",
        "n_non_dominated",
        "mean_policy_id",
        "tail_policy_id",
        "confirmatory_policy_id",
        "witness_count",
        "chosen_policy_exposure_count",
        "consequence_active",
        "frontier_warning_active",
        "primary_regime",
        "trust_state",
    ),
)

COMMON_SUMMARY_COLUMNS = (
    "benchmark_name",
    "family_name",
    "score_name",
    "n_feasible",
    "n_non_dominated",
    "witness_count",
    "witness_policy_pair_count",
    "chosen_policy_exposure_count",
    "chosen_policy_exposure_policy_pair_count",
    "mean_policy_id",
    "tail_policy_id",
    "confirmatory_policy_id",
    "consequence_active",
    "consequence_pattern",
    "frontier_warning_active",
    "primary_regime",
    "trust_state",
    "warning_reason",
    "mean_energy_span",
    "tail_energy_span",
    "selector_disagreement_count",
)

FAMILY_LADDER_SCHEMA = ExportSchema(
    name="family_ladder_summary",
    columns=COMMON_SUMMARY_COLUMNS,
)

SCORE_COMPARISON_SCHEMA = ExportSchema(
    name="score_comparison_summary",
    columns=COMMON_SUMMARY_COLUMNS,
)

NESTED_MENU_SCHEMA = ExportSchema(
    name="nested_menu_summary",
    columns=("menu_size",) + COMMON_SUMMARY_COLUMNS,
)

FRONTIER_PANEL_SCHEMA = ExportSchema(
    name="frontier_panel_summary",
    columns=("split_seed",) + COMMON_SUMMARY_COLUMNS,
)

FRONTIER_CHARACTERIZATION_SCHEMA = ExportSchema(
    name="frontier_panel_characterization",
    columns=(
        "reference_split_seed",
        "n_panel_splits",
        "n_alternative_splits",
        "reference_primary_regime",
        "modal_alternative_primary_regime",
        "reference_frontier_warning_active",
        "reference_witness_row_count",
        "reference_exposure_row_count",
        "alternative_regime_change_rate",
        "alternative_consequence_change_rate",
        "alternative_frontier_warning_activation_rate",
        "alternative_frontier_warning_change_rate",
        "alternative_mean_policy_change_rate",
        "alternative_tail_policy_change_rate",
        "alternative_confirmatory_policy_change_rate",
        "mean_feasible_jaccard",
        "median_feasible_jaccard",
        "min_feasible_jaccard",
        "max_feasible_jaccard",
        "mean_witness_count_ratio",
        "mean_exposure_count_ratio",
        "alternative_witness_zero_to_positive_rate",
        "alternative_exposure_zero_to_positive_rate",
    ),
)

CORROBORATION_SCHEMA = ExportSchema(
    name="corroboration_summary",
    columns=COMMON_SUMMARY_COLUMNS,
)


def build_frame(rows: Sequence[Mapping[str, Any]], schema: ExportSchema) -> pd.DataFrame:
    normalized_rows = []
    for row in rows:
        normalized_rows.append({column: row.get(column) for column in schema.columns})
    return pd.DataFrame(normalized_rows, columns=list(schema.columns))


def ensure_columns(frame: pd.DataFrame, schema: ExportSchema) -> pd.DataFrame:
    for column in schema.columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(schema.columns)]


SCHEMA_REGISTRY = {
    schema.name: schema
    for schema in [
        AUDIT_CARD_SCHEMA,
        FAMILY_LADDER_SCHEMA,
        SCORE_COMPARISON_SCHEMA,
        NESTED_MENU_SCHEMA,
        FRONTIER_PANEL_SCHEMA,
        FRONTIER_CHARACTERIZATION_SCHEMA,
        CORROBORATION_SCHEMA,
    ]
}


def get_schema(name: str) -> ExportSchema:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown export schema: {name}")
    return SCHEMA_REGISTRY[name]


def list_schemas() -> list[dict[str, Any]]:
    return [{"name": schema.name, "columns": list(schema.columns)} for _, schema in sorted(SCHEMA_REGISTRY.items())]
