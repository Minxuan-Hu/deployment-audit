from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from deployment_audit.export.schema import AUDIT_CARD_SCHEMA, ensure_columns


def export_audit_card(record_path: str | Path, output_path: str | Path) -> Path:
    record_path = Path(record_path)
    output_path = Path(output_path)
    data = json.loads(record_path.read_text(encoding="utf-8"))
    frame = ensure_columns(pd.DataFrame([data["audit_card"]]), AUDIT_CARD_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
