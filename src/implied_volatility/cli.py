"""Command-line entrypoint: ``ivol run``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ivol", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the leakage-free IV prediction pipeline")
    run_p.add_argument("--csv", type=Path, default=None, help="Path to spy_2020_2022.csv")
    run_p.add_argument("--sample-size", type=int, default=200_000)
    run_p.add_argument("--test-fraction", type=float, default=0.2)
    run_p.add_argument("--risk-free-rate", type=float, default=0.01)
    run_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = Config(
            sample_size=args.sample_size,
            test_fraction=args.test_fraction,
            risk_free_rate=args.risk_free_rate,
        )
        run(config, csv_path=args.csv, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
