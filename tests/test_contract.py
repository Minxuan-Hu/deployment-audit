import pandas as pd

from deployment_audit.evaluation.admissibility import evaluate_admissibility
from deployment_audit.evaluation.contract import RiskCoverageContract


def test_admissibility_respects_contract_thresholds() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "example_id": f"ex-{index}",
                "policy_id": "p1",
                "family_name": "family",
                "score_name": "bin_margin",
                "split": "calibration" if index < 10 else "test",
                "accepted": True,
                "correct": index not in {2, 5, 7},
                "energy_joules": 1.0,
                "tokens": 10.0,
                "latency_ms": 5.0,
            }
        )
    df = pd.DataFrame(rows)
    contract = RiskCoverageContract(target_risk=0.20, target_coverage=0.40, min_accepted_calibration=5)
    summary = evaluate_admissibility(df, contract)
    assert bool(summary.iloc[0]["feasible"]) is False
