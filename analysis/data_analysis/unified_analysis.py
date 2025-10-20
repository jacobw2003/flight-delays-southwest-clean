#!/usr/bin/env python3
"""
Unified Analysis Entrypoint

This module consolidates all analysis tasks into a single script.
It orchestrates the comprehensive analysis, including route, seasonal,
and STL decomposition analyses, and saves all visualizations.
"""

from pathlib import Path
import sys

# Ensure current directory is on sys.path for sibling imports when executed directly
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from comprehensive_data_analysis import SouthwestDataAnalyzer  # noqa: E402


def run_all(data_path: str | None = None) -> bool:
    analyzer = SouthwestDataAnalyzer(data_path=data_path)
    success = analyzer.run_complete_analysis()
    return bool(success)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run unified analysis pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to CSV dataset (defaults to project preprocessed data)",
    )
    args = parser.parse_args()

    ok = run_all(data_path=args.data_path)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()



