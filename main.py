"""
Main execution script for the Credit Scorecard Pipeline.
Provides command-line interface (CLI) to configure and run the scorecard pipeline.
"""

import argparse
import sys
from config.settings import ScorecardConfig
from src.pipeline import ScorecardPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regulatory-grade Credit Scorecard Development Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ac.gz",
        help="Path to raw loan dataset file (.csv or .csv.gz / .gz)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row limit to sample for fast prototyping/testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where models, tables, and figures will be saved",
    )
    parser.add_argument(
        "--pdo",
        type=float,
        default=50.0,
        help="Points to Double the Odds (default: 50.0)",
    )
    parser.add_argument(
        "--base-score",
        type=float,
        default=600.0,
        help="Base Score benchmark (default: 600.0)",
    )
    parser.add_argument(
        "--base-odds",
        type=float,
        default=50.0,
        help="Target Good:Bad odds at Base Score (default: 50.0)",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Disable saving diagnostic and calibration plots to disk",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display interactive matplotlib plot windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ScorecardConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        pdo=args.pdo,
        base_score=args.base_score,
        base_odds=args.base_odds,
    )

    pipeline = ScorecardPipeline(config=config)

    try:
        pipeline.run(
            filepath=args.data,
            sample_size=args.sample_size,
            save_plots=not args.no_save_plots,
            show_plots=args.show_plots,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        print("Please ensure the dataset file exists or specify --data <path>", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
