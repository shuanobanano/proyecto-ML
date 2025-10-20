"""Feature engineering using PySpark."""
from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .paths import SPARK_DATA_PATH

LOGGER = logging.getLogger(__name__)


def _build_spark(app_name: str = "props_spark") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )


def transform_with_spark(
    parquet_path: str,
    output_path: str = str(SPARK_DATA_PATH),
    app_name: str = "props_spark",
) -> None:
    LOGGER.info("Starting Spark session")
    spark = _build_spark(app_name=app_name)

    LOGGER.info("Reading parquet %s", parquet_path)
    df = spark.read.parquet(parquet_path)

    numeric_cols = [
        "target_price",
        "surface_total_m2",
        "surface_covered_m2",
        "rooms",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
        "price_per_m2",
        "covered_ratio",
        "rooms_per_surface",
        "property_age",
    ]

    for column in numeric_cols:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("double"))

    df = df.withColumn(
        "rooms_per_surface",
        F.when(F.col("surface_total_m2") > 0, F.col("rooms") / F.col("surface_total_m2")),
    )

    df = df.withColumn(
        "covered_ratio",
        F.when(F.col("surface_total_m2") > 0, F.col("surface_covered_m2") / F.col("surface_total_m2")),
    )

    df = df.withColumn(
        "price_per_m2",
        F.when(F.col("surface_total_m2") > 0, F.col("target_price") / F.col("surface_total_m2")),
    )

    df = df.withColumn(
        "is_house",
        F.when(F.lower(F.col("property_type")) == "casa", 1).otherwise(0),
    )

    df = df.withColumn(
        "is_apartment",
        F.when(F.lower(F.col("property_type")) == "departamento", 1).otherwise(0),
    )

    df = df.dropna(subset=["target_price", "surface_total_m2"])

    LOGGER.info("Writing Spark processed parquet to %s", output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(output_path)

    LOGGER.info("Spark transformation complete")
    spark.stop()

