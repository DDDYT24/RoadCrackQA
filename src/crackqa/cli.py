"""Command line entry point: crackqa <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crackqa.config import load_config
from crackqa.data_qa import run_qa, write_reports


def _qa(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not cfg.paths.raw.is_dir():
        print(
            f"dataset folder not found: {cfg.paths.raw.as_posix()} (see DATA.md)", file=sys.stderr
        )
        return 2
    report = run_qa(cfg)
    json_path, md_path = write_reports(report, Path(args.out) if args.out else cfg.paths.reports)
    for key, value in report["summary"].items():
        print(f"{key:52s} {value}")
    print(f"wrote {json_path.as_posix()} and {md_path.as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crackqa")
    sub = parser.add_subparsers(dest="command", required=True)

    qa = sub.add_parser("qa", help="check the raw dataset and write the QA report")
    qa.add_argument("--config", default="configs/default.yaml")
    qa.add_argument("--out", help="report folder (default: paths.reports from the config)")
    qa.set_defaults(func=_qa)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
