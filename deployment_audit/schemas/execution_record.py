from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionRecord:
    benchmark_name: str
    benchmark_version: str
    example_id: str
    split: str
    policy_id: str
    family_name: str
    score_name: str
    accepted: bool
    correct: bool
    energy_joules: float
    tokens: float
    latency_ms: float
    route_score: float
    bin_margin: float
    bin_score: float
    backend_name: str
    backend_version: str
    benchmark_manifest_hash: str
    policy_grid_hash: str
    prompt_features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
