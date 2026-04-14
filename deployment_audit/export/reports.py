from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from deployment_audit.export.schema import ExportSchema


def write_json_report(data: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_study_report(
    study_name: str,
    benchmark_source_path: str,
    benchmark_source_kind: str,
    contract_version: str,
    summary_schema: ExportSchema,
    summary_path: str,
    record_paths: list[str],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "study_name": study_name,
        "benchmark_source_kind": benchmark_source_kind,
        "benchmark_source_path": benchmark_source_path,
        "contract_version": contract_version,
        "summary_schema": summary_schema.name,
        "summary_columns": list(summary_schema.columns),
        "summary_path": summary_path,
        "record_paths": record_paths,
    }
    if benchmark_source_kind == "benchmark_manifest":
        report["benchmark_manifest_path"] = benchmark_source_path
    if benchmark_source_kind == "menu_audit_record":
        report["source_record_path"] = benchmark_source_path
    if benchmark_source_kind == "panel_manifest_index":
        report["panel_manifest_index_path"] = benchmark_source_path
    if extra_fields:
        report.update(extra_fields)
    return report
