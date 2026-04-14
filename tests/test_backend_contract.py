from pathlib import Path

import pandas as pd
import pytest

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.execution.runner import validate_backend_records
from deployment_audit.policy.grid import build_policy_grid
from deployment_audit.schemas.execution_record import ExecutionRecord


def _build_manifest(tmp_path: Path, n_examples: int = 4):
    return build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=1,
        split_seed=1,
        n_examples=n_examples,
        variable_length=True,
    )


def _build_examples(manifest):
    examples = pd.read_csv(manifest.example_manifest_path)
    split_df = pd.read_csv(manifest.split_manifest_path)
    return examples.merge(split_df, on="example_id", how="inner")


def _build_record(manifest, policy, example_id: str, split: str, backend_name: str = "mock", policy_grid_hash: str | None = None) -> ExecutionRecord:
    return ExecutionRecord(
        benchmark_name=manifest.benchmark_name,
        benchmark_version=manifest.benchmark_version,
        example_id=example_id,
        split=split,
        policy_id=policy.policy_id,
        family_name=policy.family_name,
        score_name=policy.score_name,
        accepted=True,
        correct=True,
        energy_joules=1.0,
        tokens=10.0,
        latency_ms=5.0,
        route_score=0.5,
        bin_margin=0.5,
        bin_score=0.5,
        backend_name=backend_name,
        backend_version="test",
        benchmark_manifest_hash=manifest.content_hash(),
        policy_grid_hash=policy.grid_hash if policy_grid_hash is None else policy_grid_hash,
        prompt_features={"length": 1},
    )


def _build_valid_records(manifest, policy, examples: pd.DataFrame, backend_name: str = "mock") -> list[ExecutionRecord]:
    return [
        _build_record(
            manifest,
            policy,
            example_id=str(row.example_id),
            split=str(row.split),
            backend_name=backend_name,
        )
        for row in examples[["example_id", "split"]].itertuples(index=False)
    ]


def test_validate_backend_records_rejects_count_mismatch(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=4)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    records = _build_valid_records(manifest, policy, examples)[:1]
    with pytest.raises(ValueError, match="expected"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")


def test_validate_backend_records_rejects_backend_mismatch(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=2)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    records = _build_valid_records(manifest, policy, examples, backend_name="llm")
    with pytest.raises(ValueError, match="backend_name"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")


def test_validate_backend_records_rejects_duplicate_example_id(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=3)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    rows = list(examples[["example_id", "split"]].itertuples(index=False))
    records = [
        _build_record(manifest, policy, example_id=str(rows[0].example_id), split=str(rows[0].split)),
        _build_record(manifest, policy, example_id=str(rows[0].example_id), split=str(rows[0].split)),
        _build_record(manifest, policy, example_id=str(rows[1].example_id), split=str(rows[1].split)),
    ]
    with pytest.raises(ValueError, match="duplicate example_id"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")


def test_validate_backend_records_rejects_unknown_example_id(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=2)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    records = _build_valid_records(manifest, policy, examples)
    records[-1] = _build_record(manifest, policy, example_id="unknown-example", split="test")
    with pytest.raises(ValueError, match="example_id"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")


def test_validate_backend_records_rejects_split_mismatch(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=2)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    records = _build_valid_records(manifest, policy, examples)
    first = records[0]
    wrong_split = "test" if first.split != "test" else "calibration"
    records[0] = _build_record(manifest, policy, example_id=first.example_id, split=wrong_split)
    with pytest.raises(ValueError, match="split"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")


def test_validate_backend_records_rejects_policy_grid_hash_mismatch(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, n_examples=2)
    examples = _build_examples(manifest)
    policy = build_policy_grid("length_aware_fastpath", "bin_margin")[0]
    records = _build_valid_records(manifest, policy, examples)
    first = records[0]
    records[0] = _build_record(
        manifest,
        policy,
        example_id=first.example_id,
        split=first.split,
        policy_grid_hash="wrong-grid-hash",
    )
    with pytest.raises(ValueError, match="policy_grid_hash"):
        validate_backend_records(records, manifest, policy, examples, backend_name="mock")
