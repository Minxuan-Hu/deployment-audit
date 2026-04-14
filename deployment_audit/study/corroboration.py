from __future__ import annotations

from pathlib import Path

from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.export.reports import build_study_report, write_json_report
from deployment_audit.export.schema import CORROBORATION_SCHEMA, build_frame
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.study.common import evaluate_policy_family


def run_corroboration(
    benchmark_manifest: BenchmarkManifest,
    benchmark_manifest_path: str | Path,
    backend_name: str,
    output_root: str | Path,
    contract: RiskCoverageContract,
    score_name: str = "bin_margin",
    backend_config_path: str | Path | None = None,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    summaries: list[dict[str, object]] = []
    record_paths: list[str] = []
    for family_name in ["two_action_threshold", "length_aware_fastpath"]:
        family_root = output_root / family_name
        audit_record = evaluate_policy_family(
            benchmark_manifest=benchmark_manifest,
            family_name=family_name,
            score_name=score_name,
            backend_name=backend_name,
            contract=contract,
            output_root=family_root,
            backend_config_path=backend_config_path,
        )
        summaries.append(audit_record.export_summary)
        record_paths.append(str(family_root / "menu_audit_record.json"))
    summary_df = build_frame(summaries, CORROBORATION_SCHEMA)
    summary_path = output_root / "corroboration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    report = build_study_report(
        study_name="corroboration",
        benchmark_source_path=str(benchmark_manifest_path),
        benchmark_source_kind="benchmark_manifest",
        contract_version=contract.contract_version,
        summary_schema=CORROBORATION_SCHEMA,
        summary_path=str(summary_path),
        record_paths=record_paths,
        extra_fields={"backend_name": backend_name, "backend_config_path": None if backend_config_path is None else str(backend_config_path), "score_name": score_name},
    )
    write_json_report(report, output_root / "study_report.json")
    return summaries
