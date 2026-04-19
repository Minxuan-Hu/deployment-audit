from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from deployment_audit.evaluation.selection import attach_confirmatory_score, build_selector_summary
from deployment_audit.schemas.menu_audit_record import MenuAuditRecord


def _gap_between_best_and_second(values: pd.Series) -> float | None:
    ordered_values = values.dropna().sort_values().tolist()
    if len(ordered_values) < 2:
        return None
    return float(ordered_values[1] - ordered_values[0])


def detect_frontier_warning(
    summary_df: pd.DataFrame,
    selector_summary: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    feasible_df = summary_df.loc[summary_df["feasible"]].copy()
    if feasible_df.empty:
        return True, {
            "warning_reason": "no_feasible_policies",
            "n_feasible": 0,
            "mean_energy_span": None,
            "tail_energy_span": None,
            "mean_energy_gap": None,
            "tail_energy_gap": None,
            "confirmatory_score_gap": None,
            "mean_energy_gap_ratio": None,
            "tail_energy_gap_ratio": None,
            "selector_disagreement_count": None,
        }

    selector_summary = selector_summary or build_selector_summary(feasible_df)
    scored_df = attach_confirmatory_score(feasible_df)
    n_feasible = int(len(feasible_df))
    energy_span = float(feasible_df["mean_energy"].max() - feasible_df["mean_energy"].min())
    tail_span = float(feasible_df["tail_energy"].max() - feasible_df["tail_energy"].min())
    mean_gap = _gap_between_best_and_second(feasible_df["mean_energy"])
    tail_gap = _gap_between_best_and_second(feasible_df["tail_energy"])
    confirmatory_gap = _gap_between_best_and_second(scored_df["confirmatory_score"])
    mean_gap_ratio = None if mean_gap is None else float(mean_gap / max(1e-9, energy_span))
    tail_gap_ratio = None if tail_gap is None else float(tail_gap / max(1e-9, tail_span))
    selector_disagreement_count = max(0, int(selector_summary["selector_unique_policy_count"]) - 1)

    if n_feasible <= 2:
        warning_reason = "thin_feasible_set"
        warning_active = True
    elif selector_disagreement_count > 0 and any(
        gap_ratio is not None and gap_ratio <= 0.15 for gap_ratio in [mean_gap_ratio, tail_gap_ratio]
    ):
        warning_reason = "selector_boundary_conflict"
        warning_active = True
    elif energy_span < 0.15 and tail_span < 0.25:
        warning_reason = "compressed_frontier"
        warning_active = False
    else:
        warning_reason = "stable_enough"
        warning_active = False

    return warning_active, {
        "warning_reason": warning_reason,
        "n_feasible": n_feasible,
        "mean_energy_span": energy_span,
        "tail_energy_span": tail_span,
        "mean_energy_gap": mean_gap,
        "tail_energy_gap": tail_gap,
        "confirmatory_score_gap": confirmatory_gap,
        "mean_energy_gap_ratio": mean_gap_ratio,
        "tail_energy_gap_ratio": tail_gap_ratio,
        "selector_disagreement_count": selector_disagreement_count,
    }


def characterize_frontier_panel(records: list[tuple[int, MenuAuditRecord]], reference_split_seed: int) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    record_by_seed = {split_seed: record for split_seed, record in records}
    if reference_split_seed not in record_by_seed:
        raise KeyError(f"reference split seed {reference_split_seed} not present in frontier panel")

    reference_record = record_by_seed[reference_split_seed]
    alternative_records = [(split_seed, record) for split_seed, record in records if split_seed != reference_split_seed]
    if not alternative_records:
        return {
            "reference_split_seed": reference_split_seed,
            "n_panel_splits": 1,
            "n_alternative_splits": 0,
            "reference_primary_regime": reference_record.primary_regime,
            "modal_alternative_primary_regime": None,
            "reference_frontier_warning_active": reference_record.frontier_warning_active,
            "reference_witness_row_count": reference_record.witness_summary["witness_row_count"],
            "reference_exposure_row_count": reference_record.exposure_summary["exposure_row_count"],
            "alternative_regime_change_rate": None,
            "alternative_consequence_change_rate": None,
            "alternative_frontier_warning_activation_rate": None,
            "alternative_frontier_warning_change_rate": None,
            "alternative_mean_policy_change_rate": None,
            "alternative_tail_policy_change_rate": None,
            "alternative_confirmatory_policy_change_rate": None,
            "mean_feasible_jaccard": None,
            "median_feasible_jaccard": None,
            "min_feasible_jaccard": None,
            "max_feasible_jaccard": None,
            "mean_witness_count_ratio": None,
            "mean_exposure_count_ratio": None,
            "alternative_witness_zero_to_positive_rate": None,
            "alternative_exposure_zero_to_positive_rate": None,
        }

    reference_feasible = set(reference_record.menu_state.feasible_policy_ids())
    regime_counter: Counter[str] = Counter()
    jaccards: list[float] = []
    witness_ratios: list[float] = []
    exposure_ratios: list[float] = []
    regime_change_count = 0
    consequence_change_count = 0
    frontier_warning_activation_count = 0
    frontier_warning_change_count = 0
    mean_policy_change_count = 0
    tail_policy_change_count = 0
    confirmatory_policy_change_count = 0
    witness_zero_to_positive_count = 0
    exposure_zero_to_positive_count = 0

    reference_witness_count = int(reference_record.witness_summary["witness_row_count"])
    reference_exposure_count = int(reference_record.exposure_summary["exposure_row_count"])

    for _, record in alternative_records:
        regime_counter[record.primary_regime] += 1
        if record.primary_regime != reference_record.primary_regime:
            regime_change_count += 1
        if record.consequence_active != reference_record.consequence_active:
            consequence_change_count += 1
        if record.frontier_warning_active:
            frontier_warning_activation_count += 1
        if record.frontier_warning_active != reference_record.frontier_warning_active:
            frontier_warning_change_count += 1
        if record.menu_state.mean_policy_id != reference_record.menu_state.mean_policy_id:
            mean_policy_change_count += 1
        if record.menu_state.tail_policy_id != reference_record.menu_state.tail_policy_id:
            tail_policy_change_count += 1
        if record.menu_state.confirmatory_policy_id != reference_record.menu_state.confirmatory_policy_id:
            confirmatory_policy_change_count += 1

        alternative_feasible = set(record.menu_state.feasible_policy_ids())
        union = reference_feasible | alternative_feasible
        intersection = reference_feasible & alternative_feasible
        jaccard = 1.0 if not union else len(intersection) / float(len(union))
        jaccards.append(jaccard)

        alternative_witness_count = int(record.witness_summary["witness_row_count"])
        alternative_exposure_count = int(record.exposure_summary["exposure_row_count"])
        if reference_witness_count > 0:
            witness_ratios.append(alternative_witness_count / float(reference_witness_count))
        elif alternative_witness_count > 0:
            witness_zero_to_positive_count += 1
        if reference_exposure_count > 0:
            exposure_ratios.append(alternative_exposure_count / float(reference_exposure_count))
        elif alternative_exposure_count > 0:
            exposure_zero_to_positive_count += 1

    jaccard_series = pd.Series(jaccards, dtype=float)
    witness_ratio_series = pd.Series(witness_ratios, dtype=float) if witness_ratios else None
    exposure_ratio_series = pd.Series(exposure_ratios, dtype=float) if exposure_ratios else None
    modal_alternative_primary_regime = sorted(regime for regime, count in regime_counter.items() if count == max(regime_counter.values()))[0]

    n_alternatives = len(alternative_records)
    return {
        "reference_split_seed": reference_split_seed,
        "n_panel_splits": len(records),
        "n_alternative_splits": n_alternatives,
        "reference_primary_regime": reference_record.primary_regime,
        "modal_alternative_primary_regime": modal_alternative_primary_regime,
        "reference_frontier_warning_active": reference_record.frontier_warning_active,
        "reference_witness_row_count": reference_witness_count,
        "reference_exposure_row_count": reference_exposure_count,
        "alternative_regime_change_rate": regime_change_count / float(n_alternatives),
        "alternative_consequence_change_rate": consequence_change_count / float(n_alternatives),
        "alternative_frontier_warning_activation_rate": frontier_warning_activation_count / float(n_alternatives),
        "alternative_frontier_warning_change_rate": frontier_warning_change_count / float(n_alternatives),
        "alternative_mean_policy_change_rate": mean_policy_change_count / float(n_alternatives),
        "alternative_tail_policy_change_rate": tail_policy_change_count / float(n_alternatives),
        "alternative_confirmatory_policy_change_rate": confirmatory_policy_change_count / float(n_alternatives),
        "mean_feasible_jaccard": float(jaccard_series.mean()),
        "median_feasible_jaccard": float(jaccard_series.median()),
        "min_feasible_jaccard": float(jaccard_series.min()),
        "max_feasible_jaccard": float(jaccard_series.max()),
        "mean_witness_count_ratio": None if witness_ratio_series is None else float(witness_ratio_series.mean()),
        "mean_exposure_count_ratio": None if exposure_ratio_series is None else float(exposure_ratio_series.mean()),
        "alternative_witness_zero_to_positive_rate": witness_zero_to_positive_count / float(n_alternatives),
        "alternative_exposure_zero_to_positive_rate": exposure_zero_to_positive_count / float(n_alternatives),
    }
