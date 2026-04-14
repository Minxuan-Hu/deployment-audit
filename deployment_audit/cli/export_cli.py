from __future__ import annotations

import argparse
import json
from pathlib import Path

from deployment_audit.export.figures import (
    export_family_ladder_plot_frame,
    export_frontier_characterization_plot_frame,
    export_score_comparison_plot_frame,
)
from deployment_audit.export.reports import write_json_report
from deployment_audit.export.schema import get_schema, list_schemas
from deployment_audit.export.tables import export_audit_card


def register_export_subcommands(subparsers: argparse._SubParsersAction) -> None:
    audit_parser = subparsers.add_parser("audit-card", help="Export an audit card CSV from a menu audit record")
    audit_parser.add_argument("--record", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.set_defaults(handler=_handle_audit_card)

    schema_parser = subparsers.add_parser("schema", help="Describe export schemas")
    schema_parser.add_argument("--name")
    schema_parser.set_defaults(handler=_handle_schema)

    report_parser = subparsers.add_parser("report-index", help="Build a compact report index from study reports")
    report_parser.add_argument("--study-roots", nargs="+", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(handler=_handle_report_index)

    plot_parser = subparsers.add_parser("plot-frame", help="Export a plot-ready frame from a study summary")
    plot_parser.add_argument("--kind", required=True, choices=["family-ladder", "score-comparison", "frontier-characterization"])
    plot_parser.add_argument("--source", required=True)
    plot_parser.add_argument("--output", required=True)
    plot_parser.set_defaults(handler=_handle_plot_frame)


def _handle_audit_card(args: argparse.Namespace) -> int:
    output_path = export_audit_card(record_path=args.record, output_path=args.output)
    print(output_path)
    return 0


def _handle_schema(args: argparse.Namespace) -> int:
    if args.name:
        schema = get_schema(args.name)
        print(json.dumps({"name": schema.name, "columns": list(schema.columns)}, indent=2, sort_keys=True))
        return 0
    print(json.dumps(list_schemas(), indent=2, sort_keys=True))
    return 0


def _handle_report_index(args: argparse.Namespace) -> int:
    reports = []
    for study_root in args.study_roots:
        report_path = Path(study_root) / "study_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"study_report.json not found under {study_root}")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    output_path = write_json_report({"reports": reports}, args.output)
    print(output_path)
    return 0


def _handle_plot_frame(args: argparse.Namespace) -> int:
    if args.kind == "family-ladder":
        output_path = export_family_ladder_plot_frame(summary_path=args.source, output_path=args.output)
    elif args.kind == "score-comparison":
        output_path = export_score_comparison_plot_frame(summary_path=args.source, output_path=args.output)
    else:
        output_path = export_frontier_characterization_plot_frame(characterization_path=args.source, output_path=args.output)
    print(output_path)
    return 0
