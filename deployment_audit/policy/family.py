from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    family_name: str
    family_version: str
    score_name: str
    rule: dict[str, Any]
    energy_bias: float
    token_bias: float
    latency_bias: float
    grid_hash: str

    def _acceptance_score(self, route_score: float, bin_margin: float, bin_score: float) -> float:
        if self.score_name == "route_score":
            return route_score
        if self.score_name == "bin_margin":
            return bin_margin
        if self.score_name == "bin_score":
            return bin_score
        raise KeyError(f"Unknown score_name: {self.score_name}")

    def accepts(self, pad_words: int, difficulty_tier: str, route_score: float, bin_margin: float, bin_score: float) -> bool:
        if self.family_name == "two_action_threshold":
            score_value = self._acceptance_score(route_score=route_score, bin_margin=bin_margin, bin_score=bin_score)
            return score_value >= float(self.rule["threshold"])
        if self.family_name == "length_aware_threshold":
            threshold = self.rule["long_threshold"] if pad_words >= self.rule["pad_boundary"] else self.rule["short_threshold"]
            score_value = self._acceptance_score(route_score=route_score, bin_margin=bin_margin, bin_score=bin_score)
            return score_value >= threshold
        if self.family_name == "length_aware_evidence_threshold":
            score_value = self._acceptance_score(route_score=route_score, bin_margin=bin_margin, bin_score=bin_score)
            length_threshold = self.rule["long_threshold"] if pad_words >= self.rule["pad_boundary"] else self.rule["short_threshold"]
            difficulty_threshold = self.rule["hard_bonus_threshold"] if difficulty_tier == "hard" else length_threshold
            return score_value >= difficulty_threshold
        if self.family_name == "length_aware_fastpath":
            score_value = self._acceptance_score(route_score=route_score, bin_margin=bin_margin, bin_score=bin_score)
            if pad_words >= self.rule["high_pad_boundary"]:
                threshold = self.rule["high_threshold"]
            elif pad_words >= self.rule["mid_pad_boundary"]:
                threshold = self.rule["mid_threshold"]
            else:
                threshold = self.rule["low_threshold"]
            return score_value >= threshold
        raise KeyError(f"Unknown family_name: {self.family_name}")


def _grid_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
