from __future__ import annotations

from typing import Any

from deployment_audit.schemas.menu_state import MenuState


def build_audit_card(
    menu_state: MenuState,
    witness_count: int,
    exposure_count: int,
    consequence_active: bool,
    frontier_warning_active: bool,
    primary_regime: str,
    trust_state: str,
) -> dict[str, Any]:
    return {
        "benchmark_name": menu_state.benchmark_name,
        "family_name": menu_state.family_name,
        "score_name": menu_state.score_name,
        "n_candidates": menu_state.n_candidates,
        "n_feasible": menu_state.n_feasible,
        "n_non_dominated": menu_state.n_non_dominated,
        "mean_policy_id": menu_state.mean_policy_id,
        "tail_policy_id": menu_state.tail_policy_id,
        "confirmatory_policy_id": menu_state.confirmatory_policy_id,
        "witness_count": witness_count,
        "chosen_policy_exposure_count": exposure_count,
        "consequence_active": consequence_active,
        "frontier_warning_active": frontier_warning_active,
        "primary_regime": primary_regime,
        "trust_state": trust_state,
    }
