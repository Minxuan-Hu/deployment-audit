from __future__ import annotations

from collections import Counter
from typing import Any


def extract_policy_exposure(
    witness_rows: list[dict[str, Any]],
    policy_id: str | None,
    policy_role: str,
) -> list[dict[str, Any]]:
    if policy_id is None:
        return []
    rows: list[dict[str, Any]] = []
    for witness in witness_rows:
        if witness["left_policy_id"] == policy_id or witness["right_policy_id"] == policy_id:
            rows.append(witness | {"exposure_policy_id": policy_id, "exposure_policy_role": policy_role})
    return rows


def summarize_policy_exposure(
    exposure_rows: list[dict[str, Any]],
    policy_id: str | None,
    policy_role: str,
) -> dict[str, Any]:
    proxy_counter = Counter(str(row["proxy_name"]) for row in exposure_rows)
    policy_pair_count = len({str(row["policy_pair_id"]) for row in exposure_rows})
    return {
        "exposure_policy_id": policy_id,
        "exposure_policy_role": policy_role,
        "exposure_row_count": len(exposure_rows),
        "exposure_policy_pair_count": policy_pair_count,
        "exposure_proxy_row_count_by_name": dict(sorted(proxy_counter.items())),
    }


def extract_chosen_policy_exposure(witness_rows: list[dict[str, Any]], chosen_policy_id: str | None) -> list[dict[str, Any]]:
    return extract_policy_exposure(witness_rows=witness_rows, policy_id=chosen_policy_id, policy_role="confirmatory")
