from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

import pandas as pd

DEFAULT_PROXY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("mean_tokens", "tokens"),
    ("mean_latency", "latency"),
)


def _policy_pair_id(left_policy_id: str, right_policy_id: str) -> str:
    ordered_policy_ids = sorted([left_policy_id, right_policy_id])
    return "::".join(ordered_policy_ids)


def extract_contradiction_witnesses(
    summary_df: pd.DataFrame,
    proxy_columns: Iterable[tuple[str, str]] = DEFAULT_PROXY_COLUMNS,
) -> list[dict[str, Any]]:
    feasible_df = summary_df.loc[summary_df["feasible"]].copy()
    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in combinations(feasible_df.index.tolist(), 2):
        left = feasible_df.loc[left_idx]
        right = feasible_df.loc[right_idx]
        left_policy_id = str(left["policy_id"])
        right_policy_id = str(right["policy_id"])
        energy_delta = float(left["mean_energy"] - right["mean_energy"])
        if energy_delta == 0:
            continue
        energy_sign = 1 if energy_delta > 0 else -1
        pair_id = _policy_pair_id(left_policy_id, right_policy_id)
        for proxy_column, proxy_name in proxy_columns:
            proxy_delta = float(left[proxy_column] - right[proxy_column])
            if proxy_delta == 0:
                continue
            proxy_sign = 1 if proxy_delta > 0 else -1
            if proxy_sign != energy_sign:
                rows.append(
                    {
                        "policy_pair_id": pair_id,
                        "left_policy_id": left_policy_id,
                        "right_policy_id": right_policy_id,
                        "proxy_name": proxy_name,
                        "left_mean_energy": float(left["mean_energy"]),
                        "right_mean_energy": float(right["mean_energy"]),
                        "left_proxy": float(left[proxy_column]),
                        "right_proxy": float(right[proxy_column]),
                    }
                )
    return rows


def summarize_contradiction_witnesses(witness_rows: list[dict[str, Any]]) -> dict[str, Any]:
    proxy_counter = Counter(str(row["proxy_name"]) for row in witness_rows)
    policy_pair_count = len({str(row["policy_pair_id"]) for row in witness_rows})
    return {
        "witness_row_count": len(witness_rows),
        "witness_policy_pair_count": policy_pair_count,
        "witness_proxy_names": sorted(proxy_counter.keys()),
        "witness_proxy_row_count_by_name": dict(sorted(proxy_counter.items())),
    }
