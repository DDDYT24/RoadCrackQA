"""Command line entry point: crackqa <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crackqa.config import Config, load_config


def _require_raw(cfg: Config) -> bool:
    if cfg.paths.raw.is_dir():
        return True
    print(f"dataset folder not found: {cfg.paths.raw.as_posix()} (see DATA.md)", file=sys.stderr)
    return False


def _qa(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.data_qa import run_qa, write_reports

    if not _require_raw(cfg):
        return 2
    report = run_qa(cfg)
    json_path, md_path = write_reports(report, Path(args.out) if args.out else cfg.paths.reports)
    for key, value in report["summary"].items():
        print(f"{key:52s} {value}")
    print(f"wrote {json_path.as_posix()} and {md_path.as_posix()}")
    return 0


def _tile(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.tiling import build_tile_table, write_rows

    if not _require_raw(cfg):
        return 2
    rows = build_tile_table(cfg)
    path = write_rows(rows, cfg.paths.processed / "tiles.csv")
    print(f"wrote {len(rows)} tiles to {path.as_posix()}")
    return 0


def _split(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.split import make_split

    result = make_split(cfg, cfg.paths.processed / "tiles.csv")
    for split, b in result["balance"].items():
        print(split, b)
    print(f"test lock {result['test_lock']}")
    return 0


def _baseline(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.baseline import tune_on_val

    r = tune_on_val(cfg)
    print(
        f"val: threshold {r['threshold']:.4f}  precision {r['precision']:.3f}  "
        f"recall {r['recall']:.3f}  F1 {r['f1']:.3f}  PR-AUC {r['pr_auc']:.3f}"
    )
    return 0


def _train(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.train import train

    r = train(cfg)
    v = r["val"]
    print(
        f"best {r['best']}: val F1 {v['f1']:.3f} (precision {v['precision']:.3f}, recall "
        f"{v['recall']:.3f}) at threshold {r['threshold']:.3f}; train F1 "
        f"{r['train_fit']['f1']:.3f}; {r['runtime_s']} s; {r['model_version']}"
    )
    return 0


def _eval(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.evaluate import FinalRunRefused, evaluate

    try:
        report = evaluate(cfg, args.split, args.final, Path(args.config))
    except FinalRunRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    for name, m in report["results"].items():
        print(
            f"{name:16s} precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}"
        )
    return 0


def _batch(args: argparse.Namespace, cfg: Config) -> int:
    from crackqa.batch import run_batch

    if not Path(args.input).is_dir():
        print(f"input folder not found: {args.input}", file=sys.stderr)
        return 2
    s = run_batch(cfg, Path(args.input), Path(args.output), args.include)
    print(
        f"{s['ok']} ok, {s['corrupt']} corrupt, {s['skipped']} skipped, {s['flagged']} flagged; "
        f"{s['runtime_s']} s ({s['images_per_s']} images/s)"
    )
    return 0


COMMANDS = {
    "qa": (_qa, "check the raw dataset and write the QA report"),
    "tile": (_tile, "cut every split into tiles and label them from the masks"),
    "split": (_split, "write the split manifests, balance report and test lock"),
    "baseline": (_baseline, "tune the OpenCV baseline threshold on val"),
    "train": (_train, "train the ResNet18 tile classifier"),
    "eval": (_eval, "compare the model with the baselines on val, or once on test"),
    "batch": (_batch, "score a folder of photos: results.csv, overlays, summary.json"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crackqa")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (func, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.yaml")
        if name == "qa":
            p.add_argument("--out", help="report folder (default: paths.reports from the config)")
        if name == "eval":
            p.add_argument("--split", choices=["val", "test"], default="val")
            p.add_argument("--final", action="store_true", help="the single test run")
        if name == "batch":
            p.add_argument("--input", required=True, help="folder of photos")
            p.add_argument("--output", required=True, help="folder for the results")
            p.add_argument("--include", default="*", help="file name pattern, any case")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
