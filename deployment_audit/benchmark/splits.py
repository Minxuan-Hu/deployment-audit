from __future__ import annotations

from pathlib import Path
import random

import pandas as pd


def build_split_table(example_ids: list[str], split_seed: int, calibration_fraction: float = 0.2, test_fraction: float = 0.4) -> pd.DataFrame:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if calibration_fraction + test_fraction >= 1:
        raise ValueError("calibration_fraction + test_fraction must be less than 1")
    ids = list(example_ids)
    rng = random.Random(split_seed)
    rng.shuffle(ids)
    n_examples = len(ids)
    n_cal = int(round(n_examples * calibration_fraction))
    n_test = int(round(n_examples * test_fraction))
    n_fit = n_examples - n_cal - n_test
    split_rows: list[dict[str, str]] = []
    for position, example_id in enumerate(ids):
        if position < n_fit:
            split = "fit"
        elif position < n_fit + n_cal:
            split = "calibration"
        else:
            split = "test"
        split_rows.append({"example_id": example_id, "split": split})
    return pd.DataFrame(split_rows)


def write_split_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
