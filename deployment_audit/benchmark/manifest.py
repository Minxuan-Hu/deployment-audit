from __future__ import annotations

from pathlib import Path
import json

from deployment_audit.benchmark.generator import build_example_table, write_example_manifest
from deployment_audit.benchmark.registry import get_benchmark_definition, get_generator_definition
from deployment_audit.benchmark.splits import build_split_table, write_split_manifest
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.panel_manifest_index import PanelManifestEntry, PanelManifestIndex


def build_benchmark_manifest(
    output_root: str | Path,
    benchmark_name: str,
    task_name: str,
    operation_name: str,
    generator_version: str,
    data_seed: int,
    split_seed: int,
    n_examples: int,
    variable_length: bool = True,
    benchmark_version: str = "v1",
) -> BenchmarkManifest:
    output_root = Path(output_root)
    generator_definition = get_generator_definition(f"arithmetic_{operation_name}", generator_version)
    example_path = output_root / "example_manifest.csv"
    split_path = output_root / "split_manifest.csv"
    example_df = build_example_table(
        benchmark_name=benchmark_name,
        operation_name=operation_name,
        generator_version=generator_version,
        data_seed=data_seed,
        n_examples=n_examples,
        variable_length=variable_length,
    )
    split_df = build_split_table(example_ids=example_df["example_id"].tolist(), split_seed=split_seed)
    write_example_manifest(example_df, example_path)
    write_split_manifest(split_df, split_path)
    return BenchmarkManifest(
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        task_name=task_name,
        operation_name=operation_name,
        generator_name=generator_definition.generator_name,
        generator_version=generator_version,
        generator_parameters={"variable_length": variable_length},
        data_seed=data_seed,
        split_seed=split_seed,
        n_examples=n_examples,
        example_manifest_path=str(example_path),
        split_manifest_path=str(split_path),
    )


def build_registered_benchmark_manifest(
    output_root: str | Path,
    benchmark_name: str,
    benchmark_version: str,
    data_seed: int,
    split_seed: int,
    n_examples: int,
) -> BenchmarkManifest:
    definition = get_benchmark_definition(benchmark_name=benchmark_name, benchmark_version=benchmark_version)
    return build_benchmark_manifest(
        output_root=output_root,
        benchmark_name=definition.benchmark_name,
        benchmark_version=definition.benchmark_version,
        task_name=definition.task_name,
        operation_name=definition.operation_name,
        generator_version=definition.generator_version,
        data_seed=data_seed,
        split_seed=split_seed,
        n_examples=n_examples,
        variable_length=definition.variable_length,
    )


def build_registered_panel_manifest_index(
    output_root: str | Path,
    benchmark_name: str,
    benchmark_version: str,
    data_seed: int,
    split_seeds: list[int],
    n_examples: int,
    reference_split_seed: int | None = None,
) -> PanelManifestIndex:
    if not split_seeds:
        raise ValueError("split_seeds must contain at least one value")
    if reference_split_seed is None:
        reference_split_seed = split_seeds[0]
    if reference_split_seed not in split_seeds:
        raise ValueError("reference_split_seed must be a member of split_seeds")
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must be unique")

    output_root = Path(output_root)
    manifest_entries: list[PanelManifestEntry] = []
    benchmark_definition = get_benchmark_definition(benchmark_name=benchmark_name, benchmark_version=benchmark_version)
    for split_seed in split_seeds:
        split_root = output_root / f"split_{split_seed}"
        manifest = build_registered_benchmark_manifest(
            output_root=split_root,
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            data_seed=data_seed,
            split_seed=split_seed,
            n_examples=n_examples,
        )
        manifest_path = split_root / "benchmark_manifest.json"
        manifest.write_json(manifest_path)
        manifest_entries.append(
            PanelManifestEntry(
                manifest_path=str(manifest_path),
                manifest_hash=manifest.content_hash(),
                split_seed=split_seed,
            )
        )
    return PanelManifestIndex(
        benchmark_name=benchmark_definition.benchmark_name,
        benchmark_version=benchmark_definition.benchmark_version,
        data_seed=data_seed,
        n_examples=n_examples,
        reference_split_seed=reference_split_seed,
        manifest_entries=manifest_entries,
    )


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("manifest_hash", None)
    return BenchmarkManifest(**data)


def load_panel_manifest_index(path: str | Path) -> PanelManifestIndex:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [PanelManifestEntry(**entry) for entry in data.pop("manifest_entries")]
    return PanelManifestIndex(manifest_entries=entries, **data)
