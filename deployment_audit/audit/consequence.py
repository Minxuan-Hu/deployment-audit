from __future__ import annotations

from typing import Any


def summarize_consequence_activation(
    mean_policy_id: str | None,
    tail_policy_id: str | None,
    confirmatory_policy_id: str | None,
) -> dict[str, Any]:
    selected_policy_ids = [
        policy_id
        for policy_id in [mean_policy_id, tail_policy_id, confirmatory_policy_id]
        if policy_id is not None
    ]
    unique_policy_ids = sorted(set(selected_policy_ids))
    if len(selected_policy_ids) < 3:
        pattern = "incomplete_selection"
    elif len(unique_policy_ids) == 1:
        pattern = "aligned_selection"
    elif mean_policy_id == tail_policy_id != confirmatory_policy_id:
        pattern = "confirmatory_divergence"
    elif mean_policy_id == confirmatory_policy_id != tail_policy_id:
        pattern = "tail_divergence"
    elif tail_policy_id == confirmatory_policy_id != mean_policy_id:
        pattern = "mean_divergence"
    else:
        pattern = "three_way_divergence"
    return {
        "consequence_active": len(unique_policy_ids) > 1,
        "consequence_pattern": pattern,
        "selector_unique_policy_count": len(unique_policy_ids),
    }


def detect_consequence_activation(mean_policy_id: str | None, tail_policy_id: str | None, confirmatory_policy_id: str | None) -> bool:
    return bool(
        summarize_consequence_activation(
            mean_policy_id=mean_policy_id,
            tail_policy_id=tail_policy_id,
            confirmatory_policy_id=confirmatory_policy_id,
        )["consequence_active"]
    )
