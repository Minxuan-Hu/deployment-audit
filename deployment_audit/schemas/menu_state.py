from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MenuState:
    benchmark_name: str
    benchmark_version: str
    family_name: str
    family_version: str
    score_name: str
    contract_version: str
    n_candidates: int
    n_feasible: int
    n_non_dominated: int
    mean_policy_id: str | None
    tail_policy_id: str | None
    confirmatory_policy_id: str | None
    candidate_summary: list[dict[str, Any]]
    feasible_summary: list[dict[str, Any]]
    non_dominated_policy_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def feasible_policy_ids(self) -> list[str]:
        return [str(row["policy_id"]) for row in self.feasible_summary]
