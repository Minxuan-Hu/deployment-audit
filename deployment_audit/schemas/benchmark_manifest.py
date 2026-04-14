from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_name: str
    benchmark_version: str
    task_name: str
    operation_name: str
    generator_name: str
    generator_version: str
    generator_parameters: dict[str, Any]
    data_seed: int
    split_seed: int
    n_examples: int
    example_manifest_path: str
    split_manifest_path: str
    backend_input_schema_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict() | {"manifest_hash": self.content_hash()}
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path
