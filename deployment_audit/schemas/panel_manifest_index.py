from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class PanelManifestEntry:
    manifest_path: str
    manifest_hash: str
    split_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelManifestIndex:
    benchmark_name: str
    benchmark_version: str
    data_seed: int
    n_examples: int
    reference_split_seed: int
    manifest_entries: list[PanelManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_entries"] = [entry.to_dict() for entry in self.manifest_entries]
        return payload

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path
