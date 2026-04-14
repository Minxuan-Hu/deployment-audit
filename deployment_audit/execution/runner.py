from __future__ import annotations

from pathlib import Path

from collections import Counter

import pandas as pd

from deployment_audit.execution.backend_llm import LLMBackend, load_llm_backend_config
from deployment_audit.execution.backend_mock import MockBackend
from deployment_audit.execution.backend_protocol import BackendProtocol
from deployment_audit.execution.cache_store import write_execution_table
from deployment_audit.policy.family import PolicySpec
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.execution_record import ExecutionRecord


def load_backend(name: str, backend_config_path: str | Path | None = None) -> BackendProtocol:
    if name == "mock":
        return MockBackend()
    if name == "llm":
        if backend_config_path is None:
            raise ValueError("LLM backend requires a backend configuration path.")
        return LLMBackend(load_llm_backend_config(backend_config_path))
    raise KeyError(f"Unknown backend: {name}")


def validate_backend_records(
    records: list[ExecutionRecord],
    benchmark_manifest: BenchmarkManifest,
    policy: PolicySpec,
    examples: pd.DataFrame,
    backend_name: str,
) -> None:
    expected_count = len(examples)
    if len(records) != expected_count:
        raise ValueError(
            f"Backend {backend_name} returned {len(records)} records for policy {policy.policy_id}; expected {expected_count}."
        )

    expected_example_ids = {str(example_id) for example_id in examples["example_id"].tolist()}
    expected_split_by_example = {str(row.example_id): str(row.split) for row in examples[["example_id", "split"]].itertuples(index=False)}
    observed_example_ids: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, ExecutionRecord):
            raise TypeError(
                f"Backend {backend_name} returned an invalid record at position {index} for policy {policy.policy_id}."
            )
        if record.benchmark_name != benchmark_manifest.benchmark_name:
            raise ValueError("ExecutionRecord benchmark_name does not match the benchmark manifest.")
        if record.benchmark_version != benchmark_manifest.benchmark_version:
            raise ValueError("ExecutionRecord benchmark_version does not match the benchmark manifest.")
        if record.policy_id != policy.policy_id:
            raise ValueError("ExecutionRecord policy_id does not match the requested policy.")
        if record.family_name != policy.family_name:
            raise ValueError("ExecutionRecord family_name does not match the requested policy family.")
        if record.score_name != policy.score_name:
            raise ValueError("ExecutionRecord score_name does not match the requested policy score.")
        if record.backend_name != backend_name:
            raise ValueError("ExecutionRecord backend_name does not match the active backend.")
        if record.benchmark_manifest_hash != benchmark_manifest.content_hash():
            raise ValueError("ExecutionRecord benchmark_manifest_hash does not match the benchmark manifest.")
        if record.policy_grid_hash != policy.grid_hash:
            raise ValueError("ExecutionRecord policy_grid_hash does not match the requested policy grid.")
        if record.example_id not in expected_example_ids:
            raise ValueError("ExecutionRecord example_id does not match the benchmark examples.")
        expected_split = expected_split_by_example[record.example_id]
        if record.split != expected_split:
            raise ValueError("ExecutionRecord split does not match the benchmark split manifest.")
        observed_example_ids.append(record.example_id)

    duplicate_example_ids = [example_id for example_id, count in Counter(observed_example_ids).items() if count > 1]
    if duplicate_example_ids:
        raise ValueError(f"Backend {backend_name} returned duplicate example_id values: {sorted(duplicate_example_ids)}")

    missing_example_ids = sorted(expected_example_ids - set(observed_example_ids))
    if missing_example_ids:
        raise ValueError(f"Backend {backend_name} omitted example_id values: {missing_example_ids}")


def run_policy_grid(benchmark_manifest: BenchmarkManifest, policy_grid: list[PolicySpec], backend_name: str, output_path: str | Path, backend_config_path: str | Path | None = None) -> pd.DataFrame:
    examples = pd.read_csv(benchmark_manifest.example_manifest_path)
    split_df = pd.read_csv(benchmark_manifest.split_manifest_path)
    examples = examples.merge(split_df, on="example_id", how="inner")
    backend = load_backend(backend_name, backend_config_path=backend_config_path)
    execution_rows: list[dict[str, object]] = []
    for policy in policy_grid:
        records = backend.run_policy(benchmark_manifest=benchmark_manifest, examples=examples, policy=policy)
        validate_backend_records(
            records=records,
            benchmark_manifest=benchmark_manifest,
            policy=policy,
            examples=examples,
            backend_name=backend_name,
        )
        for record in records:
            execution_rows.append(record.to_dict())
    df = pd.DataFrame(execution_rows)
    write_execution_table(df, output_path)
    return df
