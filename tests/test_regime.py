from deployment_audit.audit.regime import derive_regime_labels


def test_regime_labels_cover_visible_exposed_stable() -> None:
    labels = derive_regime_labels(
        witness_count=12,
        exposure_count=4,
        consequence_active=False,
        frontier_warning_active=False,
    )
    assert labels == ["visible_contradiction", "exposed_stable"]


from deployment_audit.audit.regime import derive_primary_regime


def test_primary_regime_prioritizes_frontier_warning_when_consequence_has_no_exposure() -> None:
    assert derive_primary_regime(
        witness_count=10,
        exposure_count=0,
        consequence_active=True,
        frontier_warning_active=True,
        n_feasible=12,
    ) == "frontier_warning"


def test_primary_regime_keeps_consequence_when_exposure_is_present() -> None:
    assert derive_primary_regime(
        witness_count=10,
        exposure_count=2,
        consequence_active=True,
        frontier_warning_active=True,
        n_feasible=12,
    ) == "consequence_generating"
