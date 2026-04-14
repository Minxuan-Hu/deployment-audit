from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deployment_audit.benchmark.manifest import load_benchmark_manifest
from deployment_audit.execution.runner import run_policy_grid
from deployment_audit.policy.grid import build_policy_grid, write_policy_grid


REQUIRED_EXECUTION_COLUMNS = {
    "benchmark_name",
    "benchmark_version",
    "example_id",
    "split",
    "policy_id",
    "family_name",
    "score_name",
    "accepted",
    "correct",
    "energy_joules",
    "tokens",
    "latency_ms",
    "route_score",
    "bin_margin",
    "bin_score",
    "backend_name",
    "backend_version",
    "benchmark_manifest_hash",
    "policy_grid_hash",
    "prompt_features",
}




def _validate_backend_arguments(backend_name: str, backend_config_path: str | None) -> None:
    if backend_name == "llm" and backend_config_path is None:
        raise ValueError("The llm backend requires --backend-config.")


def _validate_execution_table_structure(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_EXECUTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"execution record table is missing required columns: {missing}")

    duplicate_rows = frame.duplicated(subset=["policy_id", "example_id"], keep=False)
    if duplicate_rows.any():
        duplicates = frame.loc[duplicate_rows, ["policy_id", "example_id"]].drop_duplicates().sort_values(["policy_id", "example_id"])
        duplicate_pairs = duplicates.to_dict(orient="records")
        raise ValueError(f"execution record table contains duplicate policy/example rows: {duplicate_pairs}")

def register_execution_subcommands(subparsers: argparse._SubParsersAction) -> None:
    run_grid_parser = subparsers.add_parser("run-grid", help="Execute a policy grid on a benchmark manifest")
    run_grid_parser.add_argument("--manifest", required=True)
    run_grid_parser.add_argument("--family-name", required=True)
    run_grid_parser.add_argument("--score-name", required=True)
    run_grid_parser.add_argument("--backend", required=True, choices=["mock", "llm"])
    run_grid_parser.add_argument("--backend-config")
    run_grid_parser.add_argument("--output-root", required=True)
    run_grid_parser.set_defaults(handler=_handle_run_grid)

    validate_table_parser = subparsers.add_parser("validate-table", help="Validate execution-table structure and uniqueness")
    validate_table_parser.add_argument("--execution-table", required=True)
    validate_table_parser.set_defaults(handler=_handle_validate_table)


def _handle_run_grid(args: argparse.Namespace) -> int:
    _validate_backend_arguments(args.backend, args.backend_config)
    manifest = load_benchmark_manifest(args.manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    policy_grid = build_policy_grid(family_name=args.family_name, score_name=args.score_name)
    write_policy_grid(policy_grid, output_root / "policy_grid.csv")
    run_policy_grid(
        benchmark_manifest=manifest,
        policy_grid=policy_grid,
        backend_name=args.backend,
        output_path=output_root / "execution_records.csv",
        backend_config_path=args.backend_config,
    )
    print(output_root / "execution_records.csv")
    return 0


def _handle_validate_table(args: argparse.Namespace) -> int:
    execution_table = Path(args.execution_table)
    frame = pd.read_csv(execution_table)
    _validate_execution_table_structure(frame)
    print(execution_table)
    return 0
