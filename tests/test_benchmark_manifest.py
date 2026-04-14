from pathlib import Path

from deployment_audit.benchmark.manifest import build_benchmark_manifest, load_benchmark_manifest


def test_benchmark_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=25,
        variable_length=True,
    )
    path = manifest.write_json(tmp_path / "benchmark" / "benchmark_manifest.json")
    loaded = load_benchmark_manifest(path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.content_hash() == manifest.content_hash()
