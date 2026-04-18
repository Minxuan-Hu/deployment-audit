from __future__ import annotations

import hashlib
import math

import pandas as pd

from deployment_audit.execution.backend_protocol import BackendProtocol
from deployment_audit.policy.family import PolicySpec
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.execution_record import ExecutionRecord


class MockBackend(BackendProtocol):
    name = "mock"
    version = "v1"

    def _stable_uniform(self, key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12)

    def run_policy(self, benchmark_manifest: BenchmarkManifest, examples: pd.DataFrame, policy: PolicySpec) -> list[ExecutionRecord]:
        records: list[ExecutionRecord] = []
        family_bias = {
            "two_action_threshold": -0.06,
            "length_aware_threshold": -0.01,
            "length_aware_evidence_threshold": 0.02,
            "length_aware_fastpath": 0.04,
        }[policy.family_name]
        score_bias = 0.04 if policy.score_name in {"bin_margin", "bin_score"} else 0.0
        for row in examples.itertuples(index=False):
            difficulty = (row.digits - 2) * 0.08 + (row.pad_words / 1024.0) * 0.18
            semantic_key = f"{benchmark_manifest.benchmark_name}|{policy.policy_id}|{row.example_id}"
            route_score = max(0.0, min(1.0, 0.9 - difficulty + 0.15 * self._stable_uniform(semantic_key + '|route')))
            bin_margin = max(0.0, min(1.0, 0.85 - 0.7 * difficulty + 0.2 * self._stable_uniform(semantic_key + '|margin')))
            bin_score = max(0.0, min(1.0, 0.5 * route_score + 0.5 * bin_margin))
            accepted = policy.accepts(
                pad_words=int(row.pad_words),
                difficulty_tier=str(row.difficulty_tier),
                route_score=route_score,
                bin_margin=bin_margin,
                bin_score=bin_score,
            )
            accuracy_probability = max(0.05, min(0.98, 0.82 - difficulty + family_bias + score_bias))
            accuracy_draw = self._stable_uniform(semantic_key + '|accuracy')
            correct = accuracy_draw < accuracy_probability
            energy_joules = policy.energy_bias + (row.pad_words / 48.0) + (0.8 if policy.score_name == "route_score" else 0.0)
            energy_joules += (1.2 - route_score) * 2.5 + self._stable_uniform(semantic_key + '|energy')
            tokens = policy.token_bias + row.pad_words + row.digits * 7 + self._stable_uniform(semantic_key + '|tokens') * 10
            latency_ms = policy.latency_bias + row.pad_words * 0.35 + row.digits * 4.0 + (1.1 - bin_margin) * 8.0
            records.append(
                ExecutionRecord(
                    benchmark_name=benchmark_manifest.benchmark_name,
                    benchmark_version=benchmark_manifest.benchmark_version,
                    example_id=str(row.example_id),
                    split=str(row.split),
                    policy_id=policy.policy_id,
                    family_name=policy.family_name,
                    score_name=policy.score_name,
                    accepted=accepted,
                    correct=correct,
                    energy_joules=float(energy_joules),
                    tokens=float(tokens),
                    latency_ms=float(latency_ms),
                    route_score=float(route_score),
                    bin_margin=float(bin_margin),
                    bin_score=float(bin_score),
                    backend_name=self.name,
                    backend_version=self.version,
                    benchmark_manifest_hash=benchmark_manifest.content_hash(),
                    policy_grid_hash=policy.grid_hash,
                    prompt_features={"digits": int(row.digits), "pad_words": int(row.pad_words), "difficulty_tier": str(row.difficulty_tier)},
                )
            )
        return records
