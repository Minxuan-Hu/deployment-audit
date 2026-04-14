from __future__ import annotations

from pathlib import Path

import pandas as pd

from deployment_audit.export.schema import (
    FAMILY_LADDER_SCHEMA,
    FRONTIER_CHARACTERIZATION_SCHEMA,
    SCORE_COMPARISON_SCHEMA,
    ensure_columns,
)


def export_family_ladder_plot_frame(summary_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(summary_path)
    frame = ensure_columns(frame, FAMILY_LADDER_SCHEMA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def export_score_comparison_plot_frame(summary_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(summary_path)
    frame = ensure_columns(frame, SCORE_COMPARISON_SCHEMA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def export_frontier_characterization_plot_frame(characterization_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(characterization_path)
    frame = ensure_columns(frame, FRONTIER_CHARACTERIZATION_SCHEMA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
