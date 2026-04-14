from __future__ import annotations

import argparse
import json

from deployment_audit.benchmark.manifest import (
    build_benchmark_manifest,
    build_registered_benchmark_manifest,
    build_registered_panel_manifest_index,
    load_benchmark_manifest,
    load_panel_manifest_index,
)
from deployment_audit.benchmark.registry import (
    get_benchmark_definition,
    get_generator_definition,
    list_benchmark_definitions,
    list_generator_definitions,
)


def register_benchmark_subcommands(subparsers: argparse._SubParsersAction) -> None:
    build_parser = subparsers.add_parser("build", help="Build a benchmark manifest from explicit arguments")
    build_parser.add_argument("--output-root", required=True)
    build_parser.add_argument("--benchmark-name", required=True)
    build_parser.add_argument("--benchmark-version", default="v1")
    build_parser.add_argument("--task-name", required=True)
    build_parser.add_argument("--operation-name", required=True, choices=["sum", "product"])
    build_parser.add_argument("--generator-version", required=True, default="v1")
    build_parser.add_argument("--data-seed", required=True, type=int)
    build_parser.add_argument("--split-seed", required=True, type=int)
    build_parser.add_argument("--n-examples", required=True, type=int)
    build_parser.add_argument("--fixed-length", action="store_true")
    build_parser.set_defaults(handler=_handle_build)

    build_registered_parser = subparsers.add_parser("build-registered", help="Build a benchmark manifest from the public benchmark registry")
    build_registered_parser.add_argument("--output-root", required=True)
    build_registered_parser.add_argument("--benchmark-name", required=True)
    build_registered_parser.add_argument("--benchmark-version", default="v1")
    build_registered_parser.add_argument("--data-seed", required=True, type=int)
    build_registered_parser.add_argument("--split-seed", required=True, type=int)
    build_registered_parser.add_argument("--n-examples", required=True, type=int)
    build_registered_parser.set_defaults(handler=_handle_build_registered)

    build_panel_parser = subparsers.add_parser("build-panel-registered", help="Build a registered panel manifest index and its member benchmark manifests")
    build_panel_parser.add_argument("--output-root", required=True)
    build_panel_parser.add_argument("--benchmark-name", required=True)
    build_panel_parser.add_argument("--benchmark-version", default="v1")
    build_panel_parser.add_argument("--data-seed", required=True, type=int)
    build_panel_parser.add_argument("--split-seeds", nargs="+", required=True, type=int)
    build_panel_parser.add_argument("--reference-split-seed", type=int)
    build_panel_parser.add_argument("--n-examples", required=True, type=int)
    build_panel_parser.set_defaults(handler=_handle_build_panel_registered)

    describe_parser = subparsers.add_parser("describe", help="Describe a benchmark manifest")
    describe_parser.add_argument("--manifest", required=True)
    describe_parser.set_defaults(handler=_handle_describe)

    describe_panel_parser = subparsers.add_parser("describe-panel", help="Describe a panel manifest index")
    describe_panel_parser.add_argument("--manifest-index", required=True)
    describe_panel_parser.set_defaults(handler=_handle_describe_panel)

    list_benchmarks_parser = subparsers.add_parser("list-benchmarks", help="List public benchmark definitions")
    list_benchmarks_parser.set_defaults(handler=_handle_list_benchmarks)

    list_generators_parser = subparsers.add_parser("list-generators", help="List public generator definitions")
    list_generators_parser.set_defaults(handler=_handle_list_generators)

    describe_benchmark_parser = subparsers.add_parser("describe-benchmark", help="Describe a registered benchmark definition")
    describe_benchmark_parser.add_argument("--benchmark-name", required=True)
    describe_benchmark_parser.add_argument("--benchmark-version", default="v1")
    describe_benchmark_parser.set_defaults(handler=_handle_describe_benchmark)

    describe_generator_parser = subparsers.add_parser("describe-generator", help="Describe a registered generator definition")
    describe_generator_parser.add_argument("--generator-name", required=True)
    describe_generator_parser.add_argument("--generator-version", default="v1")
    describe_generator_parser.set_defaults(handler=_handle_describe_generator)


def _handle_build(args: argparse.Namespace) -> int:
    manifest = build_benchmark_manifest(
        output_root=args.output_root,
        benchmark_name=args.benchmark_name,
        benchmark_version=args.benchmark_version,
        task_name=args.task_name,
        operation_name=args.operation_name,
        generator_version=args.generator_version,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        n_examples=args.n_examples,
        variable_length=not args.fixed_length,
    )
    manifest_path = manifest.write_json(f"{args.output_root}/benchmark_manifest.json")
    print(manifest_path)
    return 0


def _handle_build_registered(args: argparse.Namespace) -> int:
    manifest = build_registered_benchmark_manifest(
        output_root=args.output_root,
        benchmark_name=args.benchmark_name,
        benchmark_version=args.benchmark_version,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        n_examples=args.n_examples,
    )
    manifest_path = manifest.write_json(f"{args.output_root}/benchmark_manifest.json")
    print(manifest_path)
    return 0


def _handle_build_panel_registered(args: argparse.Namespace) -> int:
    panel_index = build_registered_panel_manifest_index(
        output_root=args.output_root,
        benchmark_name=args.benchmark_name,
        benchmark_version=args.benchmark_version,
        data_seed=args.data_seed,
        split_seeds=args.split_seeds,
        reference_split_seed=args.reference_split_seed,
        n_examples=args.n_examples,
    )
    panel_index_path = panel_index.write_json(f"{args.output_root}/panel_manifest_index.json")
    print(panel_index_path)
    return 0


def _handle_describe(args: argparse.Namespace) -> int:
    manifest = load_benchmark_manifest(args.manifest)
    print(json.dumps(manifest.to_dict() | {"manifest_hash": manifest.content_hash()}, indent=2, sort_keys=True))
    return 0


def _handle_describe_panel(args: argparse.Namespace) -> int:
    panel_index = load_panel_manifest_index(args.manifest_index)
    print(json.dumps(panel_index.to_dict(), indent=2, sort_keys=True))
    return 0


def _handle_list_benchmarks(args: argparse.Namespace) -> int:
    print(json.dumps(list_benchmark_definitions(), indent=2, sort_keys=True))
    return 0


def _handle_list_generators(args: argparse.Namespace) -> int:
    print(json.dumps(list_generator_definitions(), indent=2, sort_keys=True))
    return 0


def _handle_describe_benchmark(args: argparse.Namespace) -> int:
    definition = get_benchmark_definition(args.benchmark_name, args.benchmark_version)
    print(json.dumps(definition.to_dict(), indent=2, sort_keys=True))
    return 0


def _handle_describe_generator(args: argparse.Namespace) -> int:
    definition = get_generator_definition(args.generator_name, args.generator_version)
    print(json.dumps(definition.to_dict(), indent=2, sort_keys=True))
    return 0
