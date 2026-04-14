from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from deployment_audit.audit.record import build_menu_audit_record
from deployment_audit.evaluation.admissibility import evaluate_admissibility
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.execution.runner import run_policy_grid
from deployment_audit.policy.grid import build_policy_grid, write_policy_grid
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.menu_audit_record import MenuAuditRecord


def write_menu_audit_outputs(audit_record: MenuAuditRecord, output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "menu_state.json").write_text(
        json.dumps(audit_record.menu_state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "menu_audit_record.json").write_text(
        json.dumps(audit_record.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(audit_record.witness_rows).to_csv(output_root / "witnesses.csv", index=False)
    pd.DataFrame([audit_record.witness_summary]).to_csv(output_root / "witness_summary.csv", index=False)
    pd.DataFrame(audit_record.chosen_policy_exposure_rows).to_csv(output_root / "chosen_policy_exposure.csv", index=False)
    pd.DataFrame([audit_record.exposure_summary]).to_csv(output_root / "chosen_policy_exposure_summary.csv", index=False)
    pd.DataFrame([audit_record.consequence_summary]).to_csv(output_root / "consequence_summary.csv", index=False)
    pd.DataFrame([audit_record.frontier_summary]).to_csv(output_root / "frontier_summary.csv", index=False)
    pd.DataFrame([audit_record.audit_card]).to_csv(output_root / "audit_card.csv", index=False)
    pd.DataFrame([audit_record.export_summary]).to_csv(output_root / "export_summary.csv", index=False)


def evaluate_policy_family(
    benchmark_manifest: BenchmarkManifest,
    family_name: str,
    score_name: str,
    backend_name: str,
    contract: RiskCoverageContract,
    output_root: str | Path,
    backend_config_path: str | Path | None = None,
) -> MenuAuditRecord:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    policy_grid = build_policy_grid(family_name=family_name, score_name=score_name)
    write_policy_grid(policy_grid, output_root / "policy_grid.csv")
    execution_df = run_policy_grid(
        benchmark_manifest=benchmark_manifest,
        policy_grid=policy_grid,
        backend_name=backend_name,
        output_path=output_root / "execution_records.csv",
        backend_config_path=backend_config_path,
    )
    summary_df = evaluate_admissibility(execution_df=execution_df, contract=contract)
    summary_df.to_csv(output_root / "policy_summary.csv", index=False)
    audit_record = build_menu_audit_record(
        benchmark_name=benchmark_manifest.benchmark_name,
        benchmark_version=benchmark_manifest.benchmark_version,
        benchmark_manifest_hash=benchmark_manifest.content_hash(),
        family_name=family_name,
        family_version="v1",
        score_name=score_name,
        contract_version=contract.contract_version,
        summary_df=summary_df,
        exposure_policy_role="confirmatory",
    )
    write_menu_audit_outputs(audit_record=audit_record, output_root=output_root)
    return audit_record
