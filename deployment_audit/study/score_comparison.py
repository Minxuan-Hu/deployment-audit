from __future__ import annotations

from pathlib import Path

from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.export.reports import build_study_report, write_json_report
from deployment_audit.export.schema import SCORE_COMPARISON_SCHEMA, build_frame
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.study.common import evaluate_policy_family


def run_score_comparison(
    benchmark_manifest: BenchmarkManifest,
    family_name: str,
    score_names: list[str],
    backend_name: str,
    output_root: str | Path,
    contract: RiskCoverageContract,
    benchmark_manifest_path: str,
    backend_config_path: str | Path | None = None,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    record_paths: list[str] = []
    for score_name in score_names:
        score_root = output_root / score_name
        audit_record = evaluate_policy_family(
            benchmark_manifest=benchmark_manifest,
            family_name=family_name,
            score_name=score_name,
            backend_name=backend_name,
            contract=contract,
            output_root=score_root,
            backend_config_path=backend_config_path,
        )
        summaries.append(audit_record.export_summary)
        record_paths.append(str(score_root / "menu_audit_record.json"))
    summary_df = build_frame(summaries, SCORE_COMPARISON_SCHEMA)
    summary_path = output_root / "score_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    report = build_study_report(
        study_name="score_comparison",
        benchmark_source_path=str(benchmark_manifest_path),
        benchmark_source_kind="benchmark_manifest",
        contract_version=contract.contract_version,
        summary_schema=SCORE_COMPARISON_SCHEMA,
        summary_path=str(summary_path),
        record_paths=record_paths,
        extra_fields={"backend_name": backend_name, "backend_config_path": None if backend_config_path is None else str(backend_config_path), "family_name": family_name, "score_names": score_names},
    )
    write_json_report(report, output_root / "study_report.json")
    return summaries
