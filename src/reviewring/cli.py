"""ReviewRing AI CLI (spec 17.2).

All commands load a YAML config (base defaults merged with the track file),
write manifests, refuse to overwrite existing runs, and keep raw data
immutable. Run ``python -m reviewring.cli <command> --help`` for details.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reviewring.utils.runtime import deep_merge, load_config

logger = logging.getLogger("reviewring")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewring",
        description="Adaptive graph-language learning for review manipulation detection",
    )
    parser.add_argument("--config", required=True, help="path to a track YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="print resolved config and data availability")
    sub.add_parser("prepare", help="ingest raw data into canonical processed tables")

    p = sub.add_parser("split", help="create and save the fixed split")
    p.add_argument("--seed", type=int, default=None)

    sub.add_parser("simulate", help="plant controlled campaigns (Track B only)")

    sub.add_parser("features", help="compute text embeddings and history features (Track B)")
    sub.add_parser("graph", help="build the causal typed graph (Track B)")

    p = sub.add_parser("train", help="train one model (see track docs for choices)")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("calibrate", help="fit Platt calibration for a saved run")
    p.add_argument("--run", required=True)

    p = sub.add_parser("evaluate", help="frozen test evaluation for a saved run")
    p.add_argument("--run", required=True)

    p = sub.add_parser("replay", help="daily replay recovery measurement (Track B)")
    p.add_argument("--run", required=True)

    p = sub.add_parser("rings", help="candidate ring discovery + recovery evaluation (Track B)")
    p.add_argument("--run", required=True)

    p = sub.add_parser("explain", help="greedy relation masking + evidence card (Track B)")
    p.add_argument("--run", required=True)
    p.add_argument("--candidate", default=None)

    p = sub.add_parser("report", help="write markdown report and figures for a run")
    p.add_argument("--run", required=True)

    return parser


def resolve_config(path: str, base_path: str | None = None) -> dict:
    """Merge base.yaml defaults with the track config (track wins)."""
    root = Path(__file__).resolve().parents[2]
    base_file = Path(base_path) if base_path else root / "configs" / "base.yaml"
    base = load_config(base_file) if base_file.exists() else {}
    track = load_config(path)
    merged = deep_merge(base, track)
    # make all stored paths relative to the repository root (cwd)
    root_cwd = Path.cwd()
    for key, value in list(merged.get("paths", {}).items()):
        if isinstance(value, str) and not Path(value).is_absolute():
            merged["paths"][key] = str((root_cwd / value).resolve())
    return merged


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args.config)

    command = args.command
    if command == "inspect":
        import json

        print(json.dumps(config, indent=2, default=str))
        for key, value in config["paths"].items():
            exists = Path(value).exists() if isinstance(value, str) else None
            print(f"  paths.{key}: {'OK' if exists else 'MISSING'}")
        return 0

    if command == "prepare":
        if config["track"] == "yelp_static":
            from reviewring.pipelines import track_a

            track_a.prepare(config)
        else:
            from reviewring.pipelines import track_b_data

            track_b_data.prepare(config)
        return 0

    if command == "split":
        if config["track"] == "yelp_static":
            from reviewring.pipelines import track_a

            track_a.split(config)
        else:
            from reviewring.pipelines import track_b_data

            track_b_data.split(config)
        return 0

    if command == "simulate":
        if config["track"] == "yelp_static":
            raise SystemExit("simulate is Track B only")
        from reviewring.pipelines import track_b_data

        track_b_data.simulate(config)
        return 0

    if command == "features":
        from reviewring.pipelines import track_b_data

        track_b_data.features(config)
        return 0

    if command == "graph":
        from reviewring.pipelines import track_b_data

        track_b_data.graph(config)
        return 0

    if command == "train":
        if config["track"] == "yelp_static":
            from reviewring.pipelines import track_a

            track_a.train(config, args.model, args.seed)
        else:
            from reviewring.pipelines import track_b_train

            track_b_train.train(config, args.model, args.seed)
        return 0

    if command in {"calibrate", "evaluate"}:
        # Track A trains + calibrates + evaluates in one pass (frozen protocol);
        # Track B likewise. Kept as explicit no-op-with-note commands for
        # interface compatibility with the spec.
        logger.info(
            "%s is integrated into the frozen train() protocol for this implementation; "
            "run summary already contains calibrated test metrics.",
            command,
        )
        return 0

    if command == "replay":
        from reviewring.pipelines import track_b_eval

        track_b_eval.replay(config, args.run)
        return 0

    if command == "rings":
        from reviewring.pipelines import track_b_eval

        track_b_eval.rings(config, args.run)
        return 0

    if command == "explain":
        from reviewring.pipelines import track_b_eval

        track_b_eval.explain(config, args.run, args.candidate)
        return 0

    if command == "report":
        if config["track"] == "yelp_static":
            from reviewring.pipelines import report as report_mod

            report_mod.track_a_report(config, args.run)
        else:
            from reviewring.pipelines import track_b_eval

            track_b_eval.report(config, args.run)
        return 0

    parser.error(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
