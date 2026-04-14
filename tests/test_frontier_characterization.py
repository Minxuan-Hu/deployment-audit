from deployment_audit.audit.frontier import characterize_frontier_panel
from deployment_audit.schemas.menu_audit_record import MenuAuditRecord
from deployment_audit.schemas.menu_state import MenuState


def _record(
    primary_regime: str,
    trust_state: str,
    mean_policy_id: str | None,
    tail_policy_id: str | None,
    confirmatory_policy_id: str | None,
    feasible_ids: list[str],
    witness_count: int,
    exposure_count: int,
    consequence_active: bool,
    frontier_warning_active: bool,
) -> MenuAuditRecord:
    menu_state = MenuState(
        benchmark_name="benchmark",
        benchmark_version="v1",
        family_name="length_aware_fastpath",
        family_version="v1",
        score_name="bin_margin",
        contract_version="risk_coverage_v1",
        n_candidates=10,
        n_feasible=len(feasible_ids),
        n_non_dominated=len(feasible_ids),
        mean_policy_id=mean_policy_id,
        tail_policy_id=tail_policy_id,
        confirmatory_policy_id=confirmatory_policy_id,
        candidate_summary=[{"policy_id": policy_id} for policy_id in feasible_ids],
        feasible_summary=[{"policy_id": policy_id, "feasible": True} for policy_id in feasible_ids],
        non_dominated_policy_ids=feasible_ids,
    )
    return MenuAuditRecord(
        benchmark_name="benchmark",
        benchmark_version="v1",
        benchmark_manifest_hash="hash",
        family_name="length_aware_fastpath",
        family_version="v1",
        score_name="bin_margin",
        contract_version="risk_coverage_v1",
        menu_state=menu_state,
        selector_summary={
            "mean_policy_id": mean_policy_id,
            "tail_policy_id": tail_policy_id,
            "confirmatory_policy_id": confirmatory_policy_id,
            "selector_unique_policy_count": len({policy_id for policy_id in [mean_policy_id, tail_policy_id, confirmatory_policy_id] if policy_id is not None}),
            "selectors_aligned": len({policy_id for policy_id in [mean_policy_id, tail_policy_id, confirmatory_policy_id] if policy_id is not None}) <= 1,
        },
        witness_rows=[{"witness": index} for index in range(witness_count)],
        witness_summary={"witness_row_count": witness_count, "witness_policy_pair_count": max(0, witness_count // 2)},
        chosen_policy_exposure_rows=[{"exposure": index} for index in range(exposure_count)],
        exposure_summary={"exposure_row_count": exposure_count, "exposure_policy_pair_count": max(0, exposure_count // 2)},
        consequence_active=consequence_active,
        consequence_summary={"consequence_active": consequence_active, "consequence_pattern": "test"},
        frontier_warning_active=frontier_warning_active,
        frontier_summary={"warning_reason": "test"},
        primary_regime=primary_regime,
        trust_state=trust_state,
        regime_labels=[primary_regime],
        audit_card={"primary_regime": primary_regime, "trust_state": trust_state},
        export_summary={"primary_regime": primary_regime, "trust_state": trust_state},
    )


def test_frontier_characterization_aggregates_alternative_behavior() -> None:
    reference = _record("exposed_stable", "monitor", "p1", "p1", "p1", ["p1", "p2", "p3"], 10, 4, False, False)
    alt_one = _record("frontier_warning", "escalate", "p2", "p2", "p2", ["p2"], 2, 1, False, True)
    alt_two = _record("exposed_stable", "monitor", "p1", "p3", "p3", ["p1", "p3"], 5, 2, True, False)
    characterization = characterize_frontier_panel([(20260322, reference), (1, alt_one), (7, alt_two)], reference_split_seed=20260322)
    assert characterization["reference_primary_regime"] == "exposed_stable"
    assert characterization["modal_alternative_primary_regime"] in {"exposed_stable", "frontier_warning"}
    assert characterization["alternative_regime_change_rate"] == 0.5
    assert characterization["alternative_frontier_warning_activation_rate"] == 0.5
    assert characterization["alternative_frontier_warning_change_rate"] == 0.5
    assert characterization["reference_witness_row_count"] == 10
    assert 0.0 <= characterization["mean_feasible_jaccard"] <= 1.0
