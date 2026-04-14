from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

from deployment_audit.schemas.menu_state import MenuState


@dataclass(frozen=True)
class MenuAuditRecord:
    benchmark_name: str
    benchmark_version: str
    benchmark_manifest_hash: str
    family_name: str
    family_version: str
    score_name: str
    contract_version: str
    menu_state: MenuState
    selector_summary: dict[str, Any]
    witness_rows: list[dict[str, Any]]
    witness_summary: dict[str, Any]
    chosen_policy_exposure_rows: list[dict[str, Any]]
    exposure_summary: dict[str, Any]
    consequence_active: bool
    consequence_summary: dict[str, Any]
    frontier_warning_active: bool
    frontier_summary: dict[str, Any]
    primary_regime: str
    trust_state: str
    regime_labels: list[str]
    audit_card: dict[str, Any]
    export_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["menu_state"] = self.menu_state.to_dict()
        return payload


def load_menu_audit_record(path: str | Path) -> MenuAuditRecord:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    menu_state = MenuState(**data.pop("menu_state"))
    return MenuAuditRecord(menu_state=menu_state, **data)
