from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCoverageContract:
    target_risk: float
    target_coverage: float
    min_accepted_calibration: int
    contract_version: str = "risk_coverage_v1"
