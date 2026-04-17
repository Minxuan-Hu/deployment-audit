from __future__ import annotations

import argparse

from deployment_audit.benchmark.manifest import load_benchmark_manifest
from deployment_audit.evaluation.contract import RiskCoverageContract
from deployment_audit.study.corroboration import run_corroboration
from deployment_audit.study.family_ladder import run_family_ladder
from deployment_audit.study.frontier_panel import run_frontier_panel
from deployment_audit.study.nested_menu import run_nested_menu
from deployment_audit.study.score_comparison import run_score_comparison


def register_study_subcommands(subparsers: argparse._SubParsersAction) -> None:
    family_parser = subparsers.add_parser("family-ladder", help="Run a family ladder study")
    family_parser.add_argument("--manifest", required=True)
    family_parser.add_argument("--output-root", required=True)
    family_parser.add_argument("--backend", required=True, choices=["mock", "llm"])
    family_parser.add_argument("--backend-config")
    family_parser.add_argument("--score-name", default="bin_score")
    family_parser.add_argument("--contract-version", default="risk_coverage_v1")
    family_parser.set_defaults(handler=_handle_family_ladder)

    score_parser = subparsers.add_parser("score-comparison", help="Run a score comparison study")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--output-root", required=True)
    score_parser.add_argument("--backend", required=True, choices=["mock", "llm"])
    score_parser.add_argument("--backend-config")
    score_parser.add_argument("--family-name", required=True)
    score_parser.add_argument("--score-names", nargs="+", required=True)
    score_parser.add_argument("--contract-version", default="risk_coverage_v1")
    score_parser.set_defaults(handler=_handle_score_comparison)

    nested_parser = subparsers.add_parser("nested-menu", help="Run a nested menu study from a source menu audit record")
    nested_parser.add_argument("--source-record", required=True)
    nested_parser.add_argument("--output-root", required=True)
    nested_parser.add_argument("--menu-sizes", nargs="+", required=True)
    nested_parser.set_defaults(handler=_handle_nested_menu)

    frontier_parser = subparsers.add_parser("frontier-panel", help="Run a frontier panel study")
    frontier_parser.add_argument("--manifest-index", required=True)
    frontier_parser.add_argument("--output-root", required=True)
    frontier_parser.add_argument("--family-name", required=True)
    frontier_parser.add_argument("--score-name", required=True)
    frontier_parser.add_argument("--backend", required=True, choices=["mock", "llm"])
    frontier_parser.add_argument("--backend-config")
    frontier_parser.add_argument("--contract-version", default="risk_coverage_v1")
    frontier_parser.set_defaults(handler=_handle_frontier_panel)

    corroboration_parser = subparsers.add_parser("corroboration", help="Run a corroboration study")
    corroboration_parser.add_argument("--manifest", required=True)
    corroboration_parser.add_argument("--output-root", required=True)
    corroboration_parser.add_argument("--backend", required=True, choices=["mock", "llm"])
    corroboration_parser.add_argument("--backend-config")
    corroboration_parser.add_argument("--score-name", default="bin_score")
    corroboration_parser.add_argument("--contract-version", default="risk_coverage_v1")
    corroboration_parser.set_defaults(handler=_handle_corroboration)


def _validate_backend_arguments(backend_name: str, backend_config_path: str | None) -> None:
    if backend_name == "llm" and backend_config_path is None:
        raise ValueError("The llm backend requires --backend-config.")


def _parse_menu_size(value: str) -> int | str:
    if value == "full":
        return "full"
    size = int(value)
    if size <= 0:
        raise ValueError(f"menu size must be a positive integer or 'full': {value}")
    return size


def _build_contract(contract_version: str) -> RiskCoverageContract:
    if contract_version != "risk_coverage_v1":
        raise KeyError(f"Unknown contract_version: {contract_version}")
    return RiskCoverageContract(target_risk=0.40, target_coverage=0.40, min_accepted_calibration=10, contract_version=contract_version)


def _handle_family_ladder(args: argparse.Namespace) -> int:
    _validate_backend_arguments(args.backend, args.backend_config)
    manifest = load_benchmark_manifest(args.manifest)
    contract = _build_contract(args.contract_version)
    run_family_ladder(benchmark_manifest=manifest, backend_name=args.backend, output_root=args.output_root, contract=contract, benchmark_manifest_path=args.manifest, score_name=args.score_name, backend_config_path=args.backend_config)
    return 0


def _handle_score_comparison(args: argparse.Namespace) -> int:
    _validate_backend_arguments(args.backend, args.backend_config)
    manifest = load_benchmark_manifest(args.manifest)
    contract = _build_contract(args.contract_version)
    run_score_comparison(
        benchmark_manifest=manifest,
        family_name=args.family_name,
        score_names=args.score_names,
        backend_name=args.backend,
        output_root=args.output_root,
        contract=contract,
        benchmark_manifest_path=args.manifest,
        backend_config_path=args.backend_config,
    )
    return 0


def _handle_nested_menu(args: argparse.Namespace) -> int:
    parsed_sizes: list[int | str] = [_parse_menu_size(value) for value in args.menu_sizes]
    run_nested_menu(
        source_record_path=args.source_record,
        output_root=args.output_root,
        menu_sizes=parsed_sizes,
    )
    return 0


def _handle_frontier_panel(args: argparse.Namespace) -> int:
    _validate_backend_arguments(args.backend, args.backend_config)
    contract = _build_contract(args.contract_version)
    run_frontier_panel(
        manifest_index_path=args.manifest_index,
        family_name=args.family_name,
        score_name=args.score_name,
        backend_name=args.backend,
        output_root=args.output_root,
        contract=contract,
        backend_config_path=args.backend_config,
    )
    return 0


def _handle_corroboration(args: argparse.Namespace) -> int:
    _validate_backend_arguments(args.backend, args.backend_config)
    manifest = load_benchmark_manifest(args.manifest)
    contract = _build_contract(args.contract_version)
    run_corroboration(
        benchmark_manifest=manifest,
        benchmark_manifest_path=args.manifest,
        backend_name=args.backend,
        output_root=args.output_root,
        contract=contract,
        score_name=args.score_name,
        backend_config_path=args.backend_config,
    )
    return 0
