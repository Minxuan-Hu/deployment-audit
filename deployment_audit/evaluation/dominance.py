from __future__ import annotations

import pandas as pd


def compute_non_dominated_set(summary_df: pd.DataFrame) -> list[str]:
    feasible_df = summary_df.loc[summary_df["feasible"]].copy()
    if feasible_df.empty:
        return []
    non_dominated: list[str] = []
    rows = feasible_df.to_dict(orient="records")
    for row in rows:
        dominated = False
        for other in rows:
            if other["policy_id"] == row["policy_id"]:
                continue
            no_worse = other["mean_energy"] <= row["mean_energy"] and other["tail_energy"] <= row["tail_energy"]
            strictly_better = other["mean_energy"] < row["mean_energy"] or other["tail_energy"] < row["tail_energy"]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            non_dominated.append(str(row["policy_id"]))
    return sorted(non_dominated)
