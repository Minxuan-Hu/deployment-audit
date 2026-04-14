from pathlib import Path
import json

import pandas as pd

from deployment_audit.benchmark.manifest import build_registered_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.study.corroboration import run_corroboration


def test_corroboration_runs_from_manifest(tmp_path: Path) -> None:
    manifest = build_registered_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-product-fixed",
        benchmark_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=80,
    )
    manifest_path = manifest.write_json(tmp_path / "benchmark" / "benchmark_manifest.json")
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    run_corroboration(
        benchmark_manifest=manifest,
        benchmark_manifest_path=manifest_path,
        backend_name="mock",
        output_root=tmp_path / "corroboration",
        contract=contract,
    )
    summary_df = pd.read_csv(tmp_path / "corroboration" / "corroboration_summary.csv")
    assert set(summary_df["family_name"]) == {"two_action_threshold", "length_aware_fastpath"}
    report = json.loads((tmp_path / "corroboration" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_path"] == str(manifest_path)
