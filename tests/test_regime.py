from deployment_audit.audit.regime import derive_regime_labels


def test_regime_labels_cover_visible_exposed_stable() -> None:
    labels = derive_regime_labels(
        witness_count=12,
        exposure_count=4,
        consequence_active=False,
        frontier_warning_active=False,
    )
    assert labels == ["visible_contradiction", "exposed_stable"]
