import json
from pathlib import Path

import pandas as pd

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.export.schema import AUDIT_CARD_SCHEMA, FAMILY_LADDER_SCHEMA
from deployment_audit.export.tables import export_audit_card
from deployment_audit.study.family_ladder import run_family_ladder


def test_audit_card_export_preserves_schema_and_primary_fields(tmp_path: Path) -> None:
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
    run_family_ladder(
        benchmark_manifest=manifest,
        benchmark_manifest_path=str(manifest_path),
        backend_name="mock",
        output_root=tmp_path / "family_ladder",
        contract=contract,
    )
    record_path = tmp_path / "family_ladder" / "length_aware_fastpath" / "menu_audit_record.json"
    output_path = tmp_path / "audit_card.csv"
    export_audit_card(record_path=record_path, output_path=output_path)
    frame = pd.read_csv(output_path)
    assert list(frame.columns) == list(AUDIT_CARD_SCHEMA.columns)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert frame.loc[0, "primary_regime"] == record["primary_regime"]
    assert frame.loc[0, "trust_state"] == record["trust_state"]


def test_family_ladder_summary_obeys_export_schema(tmp_path: Path) -> None:
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
    run_family_ladder(
        benchmark_manifest=manifest,
        benchmark_manifest_path=manifest_path,
        backend_name="mock",
        output_root=tmp_path / "family_ladder",
        contract=contract,
    )
    frame = pd.read_csv(tmp_path / "family_ladder" / "family_ladder_summary.csv")
    assert list(frame.columns) == list(FAMILY_LADDER_SCHEMA.columns)
    assert {"witness_policy_pair_count", "consequence_pattern", "selector_disagreement_count"}.issubset(frame.columns)
    report = json.loads((tmp_path / "family_ladder" / "study_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_source_kind"] == "benchmark_manifest"
    assert report["benchmark_source_path"] == str(manifest_path)
