from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ExampleRecord:
    example_id: str
    operation_name: str
    a: int
    b: int
    candidate_answer: int
    true_answer: int
    label: int
    digits: int
    pad_words: int
    pad_bin: int
    difficulty_tier: str


def _choose_digit_count(rng: random.Random) -> int:
    return rng.choice([2, 3, 4])


def _choose_pad_bin(rng: random.Random, variable_length: bool) -> int:
    if variable_length:
        return rng.choice([64, 128, 256, 384, 512, 768, 1024])
    return 256


def _sample_candidate_answer(rng: random.Random, true_answer: int, digits: int) -> tuple[int, int]:
    if rng.random() < 0.55:
        return true_answer, 1
    delta_scale = 10 ** max(0, digits - 2)
    magnitude = rng.choice([1, 2, 3]) * delta_scale
    sign = -1 if rng.random() < 0.5 else 1
    candidate = max(0, true_answer + sign * magnitude)
    return candidate, 0


def _difficulty_tier(digits: int, pad_bin: int) -> str:
    if digits >= 4 or pad_bin >= 768:
        return "hard"
    if digits >= 3 or pad_bin >= 384:
        return "medium"
    return "easy"


def build_example_table(
    benchmark_name: str,
    operation_name: str,
    generator_version: str,
    data_seed: int,
    n_examples: int,
    variable_length: bool,
) -> pd.DataFrame:
    rng = random.Random(data_seed)
    rows: list[dict[str, object]] = []
    for index in range(n_examples):
        digits = _choose_digit_count(rng)
        pad_bin = _choose_pad_bin(rng, variable_length=variable_length)
        lower = 10 ** (digits - 1)
        upper = (10**digits) - 1
        a = rng.randint(lower, upper)
        b = rng.randint(lower, upper)
        true_answer = a + b if operation_name == "sum" else a * b
        candidate_answer, label = _sample_candidate_answer(rng, true_answer=true_answer, digits=digits)
        difficulty_tier = _difficulty_tier(digits=digits, pad_bin=pad_bin)
        rows.append(
            {
                "example_id": f"{benchmark_name}-{index:05d}",
                "operation_name": operation_name,
                "a": a,
                "b": b,
                "candidate_answer": candidate_answer,
                "true_answer": true_answer,
                "label": label,
                "digits": digits,
                "pad_words": pad_bin,
                "pad_bin": pad_bin,
                "difficulty_tier": difficulty_tier,
                "generator_version": generator_version,
            }
        )
    return pd.DataFrame(rows)


def write_example_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
