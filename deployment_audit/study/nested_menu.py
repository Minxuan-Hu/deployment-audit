from __future__ import annotations

from pathlib import Path

import pandas as pd

from deployment_audit.audit.record import build_menu_audit_record
from deployment_audit.export.reports import build_study_report, write_json_report
from deployment_audit.export.schema import NESTED_MENU_SCHEMA, build_frame
from deployment_audit.schemas.menu_audit_record import load_menu_audit_record
from deployment_audit.study.common import write_menu_audit_outputs


def run_nested_menu(
    source_record_path: str | Path,
    output_root: str | Path,
    menu_sizes: list[int | str],
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_record = load_menu_audit_record(source_record_path)
    source_rows = source_record.menu_state.feasible_summary or source_record.menu_state.candidate_summary
    source_df = pd.DataFrame(source_rows)
    if source_df.empty:
        raise ValueError(f"No source rows available from {source_record_path}")
    source_df = source_df.sort_values(["mean_energy", "tail_energy", "policy_id"]).reset_index(drop=True)

    results: list[dict[str, object]] = []
    record_paths: list[str] = []
    for raw_menu_size in menu_sizes:
        if raw_menu_size != "full" and int(raw_menu_size) <= 0:
            raise ValueError(f"menu_sizes must contain positive integers or 'full': {raw_menu_size}")
        current_df = source_df.copy() if raw_menu_size == "full" else source_df.head(int(raw_menu_size)).copy()
        current_df["feasible"] = True
        menu_size_label = "full" if raw_menu_size == "full" else int(raw_menu_size)
        audit_record = build_menu_audit_record(
            benchmark_name=source_record.benchmark_name,
            benchmark_version=source_record.benchmark_version,
            benchmark_manifest_hash=source_record.benchmark_manifest_hash,
            family_name=source_record.family_name,
            family_version=source_record.family_version,
            score_name=source_record.score_name,
            contract_version=source_record.contract_version,
            summary_df=current_df,
            exposure_policy_role="confirmatory",
        )
        export_summary = {"menu_size": menu_size_label} | audit_record.export_summary
        menu_dir = output_root / f"menu_{menu_size_label}"
        menu_dir.mkdir(parents=True, exist_ok=True)
        write_menu_audit_outputs(audit_record=audit_record, output_root=menu_dir)
        results.append(export_summary)
        record_paths.append(str(menu_dir / "menu_audit_record.json"))

    summary_df = build_frame(results, NESTED_MENU_SCHEMA)
    summary_path = output_root / "nested_menu_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    report = build_study_report(
        study_name="nested_menu",
        benchmark_source_path=str(source_record_path),
        benchmark_source_kind="menu_audit_record",
        contract_version=source_record.contract_version,
        summary_schema=NESTED_MENU_SCHEMA,
        summary_path=str(summary_path),
        record_paths=record_paths,
        extra_fields={
            "family_name": source_record.family_name,
            "score_name": source_record.score_name,
            "menu_sizes": ["full" if size == "full" else int(size) for size in menu_sizes],
            "source_record_path": str(source_record_path),
            "source_primary_regime": source_record.primary_regime,
            "source_trust_state": source_record.trust_state,
            "source_benchmark_manifest_hash": source_record.benchmark_manifest_hash,
        },
    )
    write_json_report(report, output_root / "study_report.json")
    return results
