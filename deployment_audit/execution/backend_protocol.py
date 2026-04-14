from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from deployment_audit.policy.family import PolicySpec
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.execution_record import ExecutionRecord


class BackendProtocol(ABC):
    name: str
    version: str

    @abstractmethod
    def run_policy(self, benchmark_manifest: BenchmarkManifest, examples: pd.DataFrame, policy: PolicySpec) -> list[ExecutionRecord]:
        raise NotImplementedError
