"""Orchestrates the full ML pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import data_download, duckdb_processing, spark_processing, train_models
from .paths import (
    FEATURE_STATS_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RAW_DATA_DIR,
    SPARK_DATA_PATH,
    SQL_DATA_PATH,
)

LOGGER = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full valuation pipeline")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading the Kaggle dataset if a CSV is already available",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        help="Optional path to a CSV file to use instead of downloading from Kaggle",
    )
    parser.add_argument(
        "--duckdb-output",
        type=str,
        default=str(SQL_DATA_PATH),
        help="Path for the DuckDB cleaned parquet",
    )
    parser.add_argument(
        "--spark-output",
        type=str,
        default=str(SPARK_DATA_PATH),
        help="Path for the Spark transformed parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.raw_csv:
        raw_csv_path = Path(args.raw_csv)
    elif args.skip_download:
        candidates = sorted(RAW_DATA_DIR.glob("**/*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "--skip-download was passed but no CSV files were found in data/raw."
            )
        raw_csv_path = candidates[0]
    else:
        raw_csv_path = data_download.download_dataset()

    LOGGER.info("Using raw CSV: %s", raw_csv_path)

    duckdb_processing.clean_dataset_with_duckdb(str(raw_csv_path), output_path=args.duckdb_output)
    spark_processing.transform_with_spark(args.duckdb_output, output_path=args.spark_output)

    train_models.train_models(args.spark_output)

    LOGGER.info("Pipeline finished successfully")
    LOGGER.info("Outputs: %s, %s, %s, %s", args.duckdb_output, args.spark_output, MODEL_PATH)
    LOGGER.info("Reports: %s, %s, %s", METRICS_PATH, FEATURE_STATS_PATH, MODEL_METADATA_PATH)


if __name__ == "__main__":
    main()

