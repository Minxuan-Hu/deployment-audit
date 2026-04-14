from pathlib import Path
import json

import pandas as pd

from deployment_audit.benchmark.manifest import build_registered_panel_manifest_index
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.study.frontier_panel import run_frontier_panel


def test_frontier_panel_runs_from_manifest_index(tmp_path: Path) -> None:
    panel_index = build_registered_panel_manifest_index(
        output_root=tmp_path / "panel_benchmarks",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        data_seed=20260322,
        split_seeds=[20260322, 1],
        n_examples=80,
    )
    panel_index_path = panel_index.write_json(tmp_path / "panel_benchmarks" / "panel_manifest_index.json")
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    run_frontier_panel(
        manifest_index_path=panel_index_path,
        family_name="length_aware_fastpath",
        score_name="bin_margin",
        backend_name="mock",
        output_root=tmp_path / "frontier_panel",
        contract=contract,
    )
    summary_df = pd.read_csv(tmp_path / "frontier_panel" / "frontier_panel_summary.csv")
    assert len(summary_df) == 2
    report = json.loads((tmp_path / "frontier_panel" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_kind"] == "panel_manifest_index"
    assert report["benchmark_source_path"] == str(panel_index_path)
