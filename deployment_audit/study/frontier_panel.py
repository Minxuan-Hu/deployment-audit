from __future__ import annotations

from pathlib import Path

from deployment_audit.audit.frontier import characterize_frontier_panel
from deployment_audit.benchmark.manifest import load_benchmark_manifest, load_panel_manifest_index
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.export.reports import build_study_report, write_json_report
from deployment_audit.export.schema import FRONTIER_CHARACTERIZATION_SCHEMA, FRONTIER_PANEL_SCHEMA, build_frame
from deployment_audit.study.common import evaluate_policy_family


def run_frontier_panel(
    manifest_index_path: str | Path,
    family_name: str,
    score_name: str,
    backend_name: str,
    output_root: str | Path,
    contract: RiskCoverageContract,
    backend_config_path: str | Path | None = None,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_index = load_panel_manifest_index(manifest_index_path)
    panel_rows: list[dict[str, object]] = []
    records: list[tuple[int, object]] = []
    record_paths: list[str] = []
    manifest_paths: list[str] = []

    for entry in manifest_index.manifest_entries:
        split_seed = entry.split_seed
        panel_root = output_root / f"split_{split_seed}"
        manifest = load_benchmark_manifest(entry.manifest_path)
        audit_record = evaluate_policy_family(
            benchmark_manifest=manifest,
            family_name=family_name,
            score_name=score_name,
            backend_name=backend_name,
            contract=contract,
            output_root=panel_root / family_name,
            backend_config_path=backend_config_path,
        )
        row = audit_record.export_summary | {"split_seed": split_seed}
        panel_rows.append(row)
        records.append((split_seed, audit_record))
        record_paths.append(str(panel_root / family_name / "menu_audit_record.json"))
        manifest_paths.append(entry.manifest_path)
    panel_df = build_frame(panel_rows, FRONTIER_PANEL_SCHEMA)
    summary_path = output_root / "frontier_panel_summary.csv"
    panel_df.to_csv(summary_path, index=False)
    characterization = characterize_frontier_panel(records=records, reference_split_seed=manifest_index.reference_split_seed)
    characterization_df = build_frame([characterization], FRONTIER_CHARACTERIZATION_SCHEMA)
    characterization_path = output_root / "frontier_panel_characterization.csv"
    characterization_df.to_csv(characterization_path, index=False)
    report = build_study_report(
        study_name="frontier_panel",
        benchmark_source_path=str(manifest_index_path),
        benchmark_source_kind="panel_manifest_index",
        contract_version=contract.contract_version,
        summary_schema=FRONTIER_PANEL_SCHEMA,
        summary_path=str(summary_path),
        record_paths=record_paths,
        extra_fields={
            "characterization_path": str(characterization_path),
            "characterization_schema": FRONTIER_CHARACTERIZATION_SCHEMA.name,
            "characterization_columns": list(FRONTIER_CHARACTERIZATION_SCHEMA.columns),
            "backend_name": backend_name,
            "backend_config_path": None if backend_config_path is None else str(backend_config_path),
            "family_name": family_name,
            "score_name": score_name,
            "reference_split_seed": manifest_index.reference_split_seed,
            "panel_manifest_paths": manifest_paths,
        },
    )
    write_json_report(report, output_root / "study_report.json")
    return panel_rows
