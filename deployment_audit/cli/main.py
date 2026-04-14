from __future__ import annotations

import argparse

from deployment_audit.cli.benchmark_cli import register_benchmark_subcommands
from deployment_audit.cli.execution_cli import register_execution_subcommands
from deployment_audit.cli.export_cli import register_export_subcommands
from deployment_audit.cli.study_cli import register_study_subcommands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deployment-audit")
    subparsers = parser.add_subparsers(dest="command_group", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark commands")
    register_benchmark_subcommands(benchmark_parser.add_subparsers(dest="benchmark_command", required=True))

    execute_parser = subparsers.add_parser("execute", help="Execution commands")
    register_execution_subcommands(execute_parser.add_subparsers(dest="execution_command", required=True))

    study_parser = subparsers.add_parser("study", help="Study commands")
    register_study_subcommands(study_parser.add_subparsers(dest="study_command", required=True))

    export_parser = subparsers.add_parser("export", help="Export commands")
    register_export_subcommands(export_parser.add_subparsers(dest="export_command", required=True))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command handler registered")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
