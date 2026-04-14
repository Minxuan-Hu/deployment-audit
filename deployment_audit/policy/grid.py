from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from deployment_audit.policy.family import PolicySpec, _grid_hash


def _build_policy_specs(family_name: str, score_name: str) -> list[PolicySpec]:
    family_version = "v1"
    rows: list[PolicySpec] = []
    if family_name == "two_action_threshold":
        thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
        for threshold in thresholds:
            rule = {"threshold": threshold}
            payload = {"family_name": family_name, "score_name": score_name, "rule": rule, "family_version": family_version}
            rows.append(
                PolicySpec(
                    policy_id=f"{family_name}-{score_name}-t{int(threshold*100):02d}",
                    family_name=family_name,
                    family_version=family_version,
                    score_name=score_name,
                    rule=rule,
                    energy_bias=1.2,
                    token_bias=0.0,
                    latency_bias=3.0,
                    grid_hash=_grid_hash(payload),
                )
            )
    elif family_name == "length_aware_threshold":
        short_thresholds = [0.35, 0.45, 0.55]
        long_thresholds = [0.45, 0.55, 0.65]
        for short_threshold, long_threshold in product(short_thresholds, long_thresholds):
            rule = {"pad_boundary": 384, "short_threshold": short_threshold, "long_threshold": long_threshold}
            payload = {"family_name": family_name, "score_name": score_name, "rule": rule, "family_version": family_version}
            rows.append(
                PolicySpec(
                    policy_id=f"{family_name}-{score_name}-s{int(short_threshold*100)}-l{int(long_threshold*100)}",
                    family_name=family_name,
                    family_version=family_version,
                    score_name=score_name,
                    rule=rule,
                    energy_bias=1.0,
                    token_bias=3.0,
                    latency_bias=2.0,
                    grid_hash=_grid_hash(payload),
                )
            )
    elif family_name == "length_aware_evidence_threshold":
        short_thresholds = [0.30, 0.40, 0.50]
        long_thresholds = [0.40, 0.50, 0.60]
        hard_thresholds = [0.45, 0.55]
        for short_threshold, long_threshold, hard_bonus_threshold in product(short_thresholds, long_thresholds, hard_thresholds):
            rule = {
                "pad_boundary": 384,
                "short_threshold": short_threshold,
                "long_threshold": long_threshold,
                "hard_bonus_threshold": hard_bonus_threshold,
            }
            payload = {"family_name": family_name, "score_name": score_name, "rule": rule, "family_version": family_version}
            rows.append(
                PolicySpec(
                    policy_id=f"{family_name}-{score_name}-s{int(short_threshold*100)}-l{int(long_threshold*100)}-h{int(hard_bonus_threshold*100)}",
                    family_name=family_name,
                    family_version=family_version,
                    score_name=score_name,
                    rule=rule,
                    energy_bias=0.8,
                    token_bias=8.0,
                    latency_bias=1.0,
                    grid_hash=_grid_hash(payload),
                )
            )
    elif family_name == "length_aware_fastpath":
        low_thresholds = [0.25, 0.35, 0.45]
        mid_thresholds = [0.35, 0.45, 0.55]
        high_thresholds = [0.45, 0.55, 0.65]
        for low_threshold, mid_threshold, high_threshold in product(low_thresholds, mid_thresholds, high_thresholds):
            rule = {
                "mid_pad_boundary": 256,
                "high_pad_boundary": 768,
                "low_threshold": low_threshold,
                "mid_threshold": mid_threshold,
                "high_threshold": high_threshold,
            }
            payload = {"family_name": family_name, "score_name": score_name, "rule": rule, "family_version": family_version}
            rows.append(
                PolicySpec(
                    policy_id=f"{family_name}-{score_name}-lo{int(low_threshold*100)}-mi{int(mid_threshold*100)}-hi{int(high_threshold*100)}",
                    family_name=family_name,
                    family_version=family_version,
                    score_name=score_name,
                    rule=rule,
                    energy_bias=0.5,
                    token_bias=12.0,
                    latency_bias=0.0,
                    grid_hash=_grid_hash(payload),
                )
            )
    else:
        raise KeyError(f"Unknown family_name: {family_name}")
    return rows


def build_policy_grid(family_name: str, score_name: str) -> list[PolicySpec]:
    return _build_policy_specs(family_name=family_name, score_name=score_name)


def write_policy_grid(policy_grid: list[PolicySpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in policy_grid:
        rows.append(
            {
                "policy_id": policy.policy_id,
                "family_name": policy.family_name,
                "family_version": policy.family_version,
                "score_name": policy.score_name,
                "rule": policy.rule,
                "energy_bias": policy.energy_bias,
                "token_bias": policy.token_bias,
                "latency_bias": policy.latency_bias,
                "grid_hash": policy.grid_hash,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
