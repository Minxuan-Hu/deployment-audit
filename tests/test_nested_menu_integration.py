from pathlib import Path
import json

import pandas as pd
import pytest

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.evaluation.selection import select_confirmatory_policy
from deployment_audit.study.common import evaluate_policy_family
from deployment_audit.study.nested_menu import run_nested_menu


def test_nested_menu_runs_from_source_record(tmp_path: Path) -> None:
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
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    evaluate_policy_family(
        benchmark_manifest=manifest,
        family_name="length_aware_fastpath",
        score_name="bin_margin",
        backend_name="mock",
        contract=contract,
        output_root=tmp_path / "source_family",
    )
    run_nested_menu(
        source_record_path=tmp_path / "source_family" / "menu_audit_record.json",
        output_root=tmp_path / "nested_menu",
        menu_sizes=[3, "full"],
    )
    summary_df = pd.read_csv(tmp_path / "nested_menu" / "nested_menu_summary.csv")
    assert len(summary_df) == 2
    report = json.loads((tmp_path / "nested_menu" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_kind"] == "menu_audit_record"
    assert report["source_record_path"] == str(tmp_path / "source_family" / "menu_audit_record.json")


def test_nested_menu_uses_shared_confirmatory_selector(tmp_path: Path) -> None:
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
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    audit_record = evaluate_policy_family(
        benchmark_manifest=manifest,
        family_name="length_aware_fastpath",
        score_name="bin_margin",
        backend_name="mock",
        contract=contract,
        output_root=tmp_path / "source_family",
    )
    run_nested_menu(
        source_record_path=tmp_path / "source_family" / "menu_audit_record.json",
        output_root=tmp_path / "nested_menu",
        menu_sizes=[3],
    )
    source_rows = audit_record.menu_state.feasible_summary or audit_record.menu_state.candidate_summary
    source_df = pd.DataFrame(source_rows).sort_values(["mean_energy", "tail_energy", "policy_id"]).reset_index(drop=True)
    expected_confirmatory_policy_id = select_confirmatory_policy(source_df.head(3).assign(feasible=True))
    record = json.loads((tmp_path / "nested_menu" / "menu_3" / "menu_audit_record.json").read_text(encoding="utf-8"))
    assert record["menu_state"]["confirmatory_policy_id"] == expected_confirmatory_policy_id


def test_nested_menu_rejects_non_positive_menu_sizes(tmp_path: Path) -> None:
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
    contract = RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10)
    evaluate_policy_family(
        benchmark_manifest=manifest,
        family_name="length_aware_fastpath",
        score_name="bin_margin",
        backend_name="mock",
        contract=contract,
        output_root=tmp_path / "source_family",
    )
    with pytest.raises(ValueError, match="positive integers"):
        run_nested_menu(
            source_record_path=tmp_path / "source_family" / "menu_audit_record.json",
            output_root=tmp_path / "nested_menu_invalid",
            menu_sizes=[0],
        )
    with pytest.raises(ValueError, match="positive integers"):
        run_nested_menu(
            source_record_path=tmp_path / "source_family" / "menu_audit_record.json",
            output_root=tmp_path / "nested_menu_invalid_negative",
            menu_sizes=[-1],
        )
