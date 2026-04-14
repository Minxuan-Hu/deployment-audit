from pathlib import Path
import json

import pandas as pd

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.study.score_comparison import run_score_comparison


def test_score_comparison_end_to_end(tmp_path: Path) -> None:
    manifest = build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=80,
        variable_length=True,
    )
    manifest_path = manifest.write_json(tmp_path / "benchmark" / "benchmark_manifest.json")
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    run_score_comparison(
        benchmark_manifest=manifest,
        benchmark_manifest_path=manifest_path,
        family_name="length_aware_fastpath",
        score_names=["bin_margin", "route_score"],
        backend_name="mock",
        output_root=tmp_path / "score_comparison",
        contract=contract,
    )
    summary_path = tmp_path / "score_comparison" / "score_comparison_summary.csv"
    df = pd.read_csv(summary_path)
    assert set(df["score_name"]) == {"bin_margin", "route_score"}
    report = json.loads((tmp_path / "score_comparison" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_path"] == str(manifest_path)
