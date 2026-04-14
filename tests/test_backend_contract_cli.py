from pathlib import Path

import pandas as pd
import pytest

from deployment_audit.cli.execution_cli import _validate_execution_table_structure


def _build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark_name": "benchmark",
                "benchmark_version": "v1",
                "example_id": "ex-1",
                "split": "calibration",
                "policy_id": "policy-a",
                "family_name": "family",
                "score_name": "score",
                "accepted": True,
                "correct": True,
                "energy_joules": 1.0,
                "tokens": 10.0,
                "latency_ms": 5.0,
                "route_score": 0.5,
                "bin_margin": 0.5,
                "bin_score": 0.5,
                "backend_name": "mock",
                "backend_version": "v1",
                "benchmark_manifest_hash": "hash",
                "policy_grid_hash": "grid",
                "prompt_features": "{}",
            }
        ]
    )


def test_validate_execution_table_structure_rejects_missing_columns() -> None:
    frame = _build_frame().drop(columns=["policy_grid_hash"])
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_execution_table_structure(frame)


def test_validate_execution_table_structure_rejects_duplicate_policy_example_rows() -> None:
    frame = pd.concat([_build_frame(), _build_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate policy/example rows"):
        _validate_execution_table_structure(frame)
