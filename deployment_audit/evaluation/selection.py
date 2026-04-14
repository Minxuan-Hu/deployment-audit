from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _feasible_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df.loc[summary_df["feasible"]].copy()


def _select_policy_id(feasible_df: pd.DataFrame, sort_columns: list[str]) -> str | None:
    if feasible_df.empty:
        return None
    ordered = feasible_df.sort_values(sort_columns, ascending=[True] * len(sort_columns))
    return str(ordered.iloc[0]["policy_id"])


def attach_confirmatory_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    scored_df = summary_df.copy()
    scored_df["confirmatory_score"] = math.nan
    feasible_df = _feasible_summary(scored_df)
    if feasible_df.empty:
        return scored_df
    mean_min = float(feasible_df["mean_energy"].min())
    tail_min = float(feasible_df["tail_energy"].min())
    mean_span = max(1e-9, float(feasible_df["mean_energy"].max() - mean_min))
    tail_span = max(1e-9, float(feasible_df["tail_energy"].max() - tail_min))
    confirmatory_score = (
        (feasible_df["mean_energy"] - mean_min) / mean_span + (feasible_df["tail_energy"] - tail_min) / tail_span
    ) / 2.0
    scored_df.loc[feasible_df.index, "confirmatory_score"] = confirmatory_score
    return scored_df


def select_mean_policy(summary_df: pd.DataFrame) -> str | None:
    return _select_policy_id(_feasible_summary(summary_df), ["mean_energy", "tail_energy", "policy_id"])


def select_tail_policy(summary_df: pd.DataFrame) -> str | None:
    return _select_policy_id(_feasible_summary(summary_df), ["tail_energy", "mean_energy", "policy_id"])


def select_confirmatory_policy(summary_df: pd.DataFrame) -> str | None:
    scored_df = attach_confirmatory_score(summary_df)
    return _select_policy_id(_feasible_summary(scored_df), ["confirmatory_score", "mean_energy", "tail_energy", "policy_id"])


def build_selector_summary(summary_df: pd.DataFrame) -> dict[str, Any]:
    mean_policy_id = select_mean_policy(summary_df)
    tail_policy_id = select_tail_policy(summary_df)
    confirmatory_policy_id = select_confirmatory_policy(summary_df)
    selected_policy_ids = [
        policy_id
        for policy_id in [mean_policy_id, tail_policy_id, confirmatory_policy_id]
        if policy_id is not None
    ]
    unique_policy_ids = sorted(set(selected_policy_ids))
    return {
        "mean_policy_id": mean_policy_id,
        "tail_policy_id": tail_policy_id,
        "confirmatory_policy_id": confirmatory_policy_id,
        "selector_unique_policy_count": len(unique_policy_ids),
        "selectors_aligned": len(unique_policy_ids) <= 1,
    }
