from __future__ import annotations

from typing import Any

import pandas as pd

from deployment_audit.audit.audit_card import build_audit_card
from deployment_audit.audit.consequence import summarize_consequence_activation
from deployment_audit.audit.exposure import extract_policy_exposure, summarize_policy_exposure
from deployment_audit.audit.frontier import detect_frontier_warning
from deployment_audit.audit.regime import derive_primary_regime, derive_regime_labels, derive_trust_state
from deployment_audit.audit.witnesses import extract_contradiction_witnesses, summarize_contradiction_witnesses
from deployment_audit.evaluation.dominance import compute_non_dominated_set
from deployment_audit.evaluation.selection import build_selector_summary
from deployment_audit.schemas.menu_audit_record import MenuAuditRecord
from deployment_audit.schemas.menu_state import MenuState


def build_menu_audit_record(
    benchmark_name: str,
    benchmark_version: str,
    benchmark_manifest_hash: str,
    family_name: str,
    family_version: str,
    score_name: str,
    contract_version: str,
    summary_df: pd.DataFrame,
    exposure_policy_role: str = "confirmatory",
) -> MenuAuditRecord:
    selector_summary = build_selector_summary(summary_df)
    mean_policy_id = selector_summary["mean_policy_id"]
    tail_policy_id = selector_summary["tail_policy_id"]
    confirmatory_policy_id = selector_summary["confirmatory_policy_id"]

    non_dominated_policy_ids = compute_non_dominated_set(summary_df)
    menu_state = MenuState(
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        family_name=family_name,
        family_version=family_version,
        score_name=score_name,
        contract_version=contract_version,
        n_candidates=int(len(summary_df)),
        n_feasible=int(summary_df["feasible"].sum()),
        n_non_dominated=len(non_dominated_policy_ids),
        mean_policy_id=mean_policy_id,
        tail_policy_id=tail_policy_id,
        confirmatory_policy_id=confirmatory_policy_id,
        candidate_summary=summary_df.to_dict(orient="records"),
        feasible_summary=summary_df.loc[summary_df["feasible"]].to_dict(orient="records"),
        non_dominated_policy_ids=non_dominated_policy_ids,
    )

    witness_rows = extract_contradiction_witnesses(summary_df)
    witness_summary = summarize_contradiction_witnesses(witness_rows)
    exposure_policy_id = selector_summary[f"{exposure_policy_role}_policy_id"]
    exposure_rows = extract_policy_exposure(
        witness_rows=witness_rows,
        policy_id=exposure_policy_id,
        policy_role=exposure_policy_role,
    )
    exposure_summary = summarize_policy_exposure(
        exposure_rows=exposure_rows,
        policy_id=exposure_policy_id,
        policy_role=exposure_policy_role,
    )
    consequence_summary = summarize_consequence_activation(
        mean_policy_id=mean_policy_id,
        tail_policy_id=tail_policy_id,
        confirmatory_policy_id=confirmatory_policy_id,
    )
    consequence_active = bool(consequence_summary["consequence_active"])
    frontier_warning_active, frontier_summary = detect_frontier_warning(summary_df, selector_summary=selector_summary)
    primary_regime = derive_primary_regime(
        witness_count=int(witness_summary["witness_row_count"]),
        exposure_count=int(exposure_summary["exposure_row_count"]),
        consequence_active=consequence_active,
        frontier_warning_active=frontier_warning_active,
        n_feasible=menu_state.n_feasible,
    )
    trust_state = derive_trust_state(primary_regime)
    regime_labels = derive_regime_labels(
        witness_count=int(witness_summary["witness_row_count"]),
        exposure_count=int(exposure_summary["exposure_row_count"]),
        consequence_active=consequence_active,
        frontier_warning_active=frontier_warning_active,
    )
    audit_card = build_audit_card(
        menu_state=menu_state,
        witness_count=int(witness_summary["witness_row_count"]),
        exposure_count=int(exposure_summary["exposure_row_count"]),
        consequence_active=consequence_active,
        frontier_warning_active=frontier_warning_active,
        primary_regime=primary_regime,
        trust_state=trust_state,
    )
    export_summary = {
        "benchmark_name": benchmark_name,
        "family_name": family_name,
        "score_name": score_name,
        "n_feasible": menu_state.n_feasible,
        "n_non_dominated": menu_state.n_non_dominated,
        "witness_count": int(witness_summary["witness_row_count"]),
        "witness_policy_pair_count": int(witness_summary["witness_policy_pair_count"]),
        "chosen_policy_exposure_count": int(exposure_summary["exposure_row_count"]),
        "chosen_policy_exposure_policy_pair_count": int(exposure_summary["exposure_policy_pair_count"]),
        "mean_policy_id": mean_policy_id,
        "tail_policy_id": tail_policy_id,
        "confirmatory_policy_id": confirmatory_policy_id,
        "consequence_active": consequence_active,
        "consequence_pattern": str(consequence_summary["consequence_pattern"]),
        "frontier_warning_active": frontier_warning_active,
        "primary_regime": primary_regime,
        "trust_state": trust_state,
        "regime_labels": regime_labels,
        **frontier_summary,
    }
    return MenuAuditRecord(
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        benchmark_manifest_hash=benchmark_manifest_hash,
        family_name=family_name,
        family_version=family_version,
        score_name=score_name,
        contract_version=contract_version,
        menu_state=menu_state,
        selector_summary=selector_summary,
        witness_rows=witness_rows,
        witness_summary=witness_summary,
        chosen_policy_exposure_rows=exposure_rows,
        exposure_summary=exposure_summary,
        consequence_active=consequence_active,
        consequence_summary=consequence_summary,
        frontier_warning_active=frontier_warning_active,
        frontier_summary=frontier_summary,
        primary_regime=primary_regime,
        trust_state=trust_state,
        regime_labels=regime_labels,
        audit_card=audit_card,
        export_summary=export_summary,
    )
