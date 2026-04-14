from pathlib import Path
import json

import pandas as pd

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.study.family_ladder import run_family_ladder


def test_family_ladder_end_to_end(tmp_path: Path) -> None:
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
    summaries = run_family_ladder(
        benchmark_manifest=manifest,
        benchmark_manifest_path=manifest_path,
        backend_name="mock",
        output_root=tmp_path / "family_ladder",
        contract=contract,
    )
    assert len(summaries) == 4
    summary_path = tmp_path / "family_ladder" / "family_ladder_summary.csv"
    assert summary_path.exists()
    df = pd.read_csv(summary_path)
    assert len(df) == 4
    assert set(df["family_name"]) == {
        "two_action_threshold",
        "length_aware_threshold",
        "length_aware_evidence_threshold",
        "length_aware_fastpath",
    }
    for family_name in df["family_name"]:
        record_path = tmp_path / "family_ladder" / family_name / "menu_audit_record.json"
        data = json.loads(record_path.read_text(encoding="utf-8"))
        assert data["menu_state"]["family_name"] == family_name
    report = json.loads((tmp_path / "family_ladder" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_path"] == str(manifest_path)
