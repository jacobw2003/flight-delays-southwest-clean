#!/usr/bin/env python3
"""
Unified ML Entrypoint

This module consolidates all ML tasks into a single script.
It offers three consistent modes:
 - consumer: two-stage consumer-facing pipeline
 - realistic: pre-departure features only (no leakage)
 - operational: operational insights focused pipeline
"""

from pathlib import Path
import sys

# Ensure current directory is on sys.path for sibling imports when executed directly
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from consumer_ml_pipeline import ConsumerMLPipeline  # noqa: E402
from realistic_model import RealisticSouthwestModel  # noqa: E402
from southwest_operational_ml import SouthwestOperationalML  # noqa: E402


def run_consumer(data_path: str | None = None):
    pipeline = ConsumerMLPipeline(data_path=data_path)
    return pipeline.run_complete_pipeline()


def run_realistic(data_path: str | None = None):
    analyzer = RealisticSouthwestModel(data_path=data_path)
    return analyzer.run_realistic_analysis()


def run_operational(data_path: str | None = None):
    pipeline = SouthwestOperationalML(data_path=data_path)
    return pipeline.run_complete_pipeline()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run unified ML pipelines")
    parser.add_argument(
        "mode",
        choices=["consumer", "realistic", "operational"],
        help="Which ML pipeline to run",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to CSV dataset (defaults to project preprocessed data)",
    )
    args = parser.parse_args()

    if args.mode == "consumer":
        res = run_consumer(args.data_path)
    elif args.mode == "realistic":
        res = run_realistic(args.data_path)
    else:
        res = run_operational(args.data_path)

    if res is None or res is False:
        sys.exit(1)


if __name__ == "__main__":
    main()


