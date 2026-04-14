from __future__ import annotations


def derive_regime_labels(witness_count: int, exposure_count: int, consequence_active: bool, frontier_warning_active: bool) -> list[str]:
    labels: list[str] = []
    if witness_count == 0:
        labels.append("hidden_contradiction")
    else:
        labels.append("visible_contradiction")
    if exposure_count > 0 and not consequence_active:
        labels.append("exposed_stable")
    if consequence_active:
        labels.append("consequence_generating")
    if frontier_warning_active:
        labels.append("frontier_warning")
    return labels


def derive_primary_regime(witness_count: int, exposure_count: int, consequence_active: bool, frontier_warning_active: bool, n_feasible: int) -> str:
    if n_feasible == 0:
        return "no_feasible_policies"
    if consequence_active:
        return "consequence_generating"
    if frontier_warning_active:
        return "frontier_warning"
    if exposure_count > 0:
        return "exposed_stable"
    if witness_count > 0:
        return "visible_contradiction"
    return "hidden_contradiction"


def derive_trust_state(primary_regime: str) -> str:
    if primary_regime in {"no_feasible_policies", "frontier_warning"}:
        return "escalate"
    if primary_regime == "consequence_generating":
        return "review"
    if primary_regime == "exposed_stable":
        return "monitor"
    return "accept"
