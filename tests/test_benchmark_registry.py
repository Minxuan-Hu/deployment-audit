import pytest

from pathlib import Path

from deployment_audit.benchmark.manifest import build_benchmark_manifest, build_registered_benchmark_manifest, build_registered_panel_manifest_index
from deployment_audit.benchmark.registry import get_benchmark_definition, list_benchmark_definitions


def test_registered_benchmark_build_roundtrip(tmp_path: Path) -> None:
    manifest = build_registered_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=25,
    )
    assert manifest.benchmark_name == "arithmetic-sum-varlen"
    assert manifest.generator_name == "arithmetic_sum"
    assert manifest.generator_parameters["variable_length"] is True


def test_registered_panel_manifest_index_builds_public_source_object(tmp_path: Path) -> None:
    panel_index = build_registered_panel_manifest_index(
        output_root=tmp_path / "panel_benchmarks",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        data_seed=20260322,
        split_seeds=[20260322, 20260323],
        n_examples=25,
    )
    assert panel_index.reference_split_seed == 20260322
    assert len(panel_index.manifest_entries) == 2
    assert all(Path(entry.manifest_path).exists() for entry in panel_index.manifest_entries)


def test_benchmark_registry_exposes_public_definitions() -> None:
    benchmark_names = {row["benchmark_name"] for row in list_benchmark_definitions()}
    assert {"arithmetic-sum-varlen", "arithmetic-sum-fixed", "arithmetic-product-varlen", "arithmetic-product-fixed"} <= benchmark_names
    definition = get_benchmark_definition("arithmetic-product-fixed", "v1")
    assert definition.variable_length is False


def test_registered_panel_manifest_index_rejects_duplicate_split_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split_seeds must be unique"):
        build_registered_panel_manifest_index(
            output_root=tmp_path / "panel_benchmarks",
            benchmark_name="arithmetic-sum-varlen",
            benchmark_version="v1",
            data_seed=20260322,
            split_seeds=[20260322, 20260322],
            n_examples=25,
        )


def test_explicit_benchmark_manifest_preserves_requested_version(tmp_path: Path) -> None:
    manifest = build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v2",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=25,
        variable_length=True,
    )
    assert manifest.benchmark_version == "v2"


def test_explicit_benchmark_build_cli_preserves_requested_version(tmp_path: Path) -> None:
    from argparse import Namespace

    from deployment_audit.benchmark.manifest import load_benchmark_manifest
    from deployment_audit.cli.benchmark_cli import _handle_build

    manifest_root = tmp_path / "benchmark"
    args = Namespace(
        output_root=str(manifest_root),
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v2",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=25,
        fixed_length=False,
    )
    assert _handle_build(args) == 0
    manifest = load_benchmark_manifest(manifest_root / "benchmark_manifest.json")
    assert manifest.benchmark_version == "v2"


def test_cli_parser_accepts_explicit_benchmark_version_for_build() -> None:
    from deployment_audit.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "benchmark",
        "build",
        "--output-root",
        "out",
        "--benchmark-name",
        "arithmetic-sum-varlen",
        "--benchmark-version",
        "v2",
        "--task-name",
        "verify_sum",
        "--operation-name",
        "sum",
        "--generator-version",
        "v1",
        "--data-seed",
        "1",
        "--split-seed",
        "2",
        "--n-examples",
        "3",
    ])
    assert args.benchmark_version == "v2"
