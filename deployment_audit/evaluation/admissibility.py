from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from deployment_audit.evaluation.contract import RiskCoverageContract


def _policy_summary(execution_df: pd.DataFrame, policy_id: str, contract: RiskCoverageContract) -> dict[str, Any]:
    policy_df = execution_df.loc[execution_df["policy_id"] == policy_id].copy()
    cal_df = policy_df.loc[policy_df["split"] == "calibration"]
    test_df = policy_df.loc[policy_df["split"] == "test"]
    n_candidates = int(policy_df["example_id"].nunique())
    cal_accepted = cal_df.loc[cal_df["accepted"]]
    test_accepted = test_df.loc[test_df["accepted"]]
    cal_accepted_n = int(len(cal_accepted))
    cal_coverage = float(cal_accepted_n / max(1, len(cal_df)))
    if cal_accepted_n == 0:
        cal_selective_risk = 1.0
    else:
        cal_selective_risk = float(1.0 - cal_accepted["correct"].mean())
    feasible = (
        cal_accepted_n >= contract.min_accepted_calibration
        and cal_coverage >= contract.target_coverage
        and cal_selective_risk <= contract.target_risk
        and len(test_accepted) > 0
    )
    if len(test_accepted) == 0:
        mean_energy = math.inf
        tail_energy = math.inf
        mean_tokens = math.inf
        mean_latency = math.inf
        test_accuracy = 0.0
    else:
        mean_energy = float(test_accepted["energy_joules"].mean())
        tail_energy = float(test_accepted["energy_joules"].quantile(0.95))
        mean_tokens = float(test_accepted["tokens"].mean())
        mean_latency = float(test_accepted["latency_ms"].mean())
        test_accuracy = float(test_accepted["correct"].mean())
    return {
        "policy_id": policy_id,
        "family_name": str(policy_df["family_name"].iloc[0]),
        "family_version": "v1",
        "score_name": str(policy_df["score_name"].iloc[0]),
        "n_examples": n_candidates,
        "n_cal_examples": int(len(cal_df)),
        "n_test_examples": int(len(test_df)),
        "cal_accepted_n": cal_accepted_n,
        "cal_coverage": cal_coverage,
        "cal_selective_risk": cal_selective_risk,
        "feasible": bool(feasible),
        "mean_energy": mean_energy,
        "tail_energy": tail_energy,
        "mean_tokens": mean_tokens,
        "mean_latency": mean_latency,
        "test_accuracy": test_accuracy,
    }


def evaluate_admissibility(execution_df: pd.DataFrame, contract: RiskCoverageContract) -> pd.DataFrame:
    rows = [_policy_summary(execution_df=execution_df, policy_id=policy_id, contract=contract) for policy_id in sorted(execution_df["policy_id"].unique())]
    return pd.DataFrame(rows)
