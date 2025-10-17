"""Data cleaning using DuckDB SQL."""
from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, Optional

import duckdb

from .paths import DUCKDB_SUMMARY_PATH, SQL_DATA_PATH

LOGGER = logging.getLogger(__name__)


class ColumnNotFoundError(RuntimeError):
    """Raised when a required column is missing in the dataset."""


REQUIRED_FEATURES: Dict[str, Iterable[str]] = {
    "price": ("price_usd", "price", "precio", "precio_usd"),
    "surface_total": ("surface_total_in_m2", "surface_total", "superficie_total", "total_surface"),
}

OPTIONAL_FEATURES: Dict[str, Iterable[str]] = {
    "surface_covered": ("surface_covered_in_m2", "surface_covered", "superficie_cubierta"),
    "rooms": ("rooms", "ambientes"),
    "bedrooms": ("bedrooms", "dormitorios"),
    "bathrooms": ("bathrooms", "banos"),
    "property_type": ("property_type", "tipo_propiedad"),
    "state_name": ("state_name", "state", "provincia", "location"),
    "neighborhood": ("neighborhood", "barrio", "place_name", "l3"),
    "currency": ("currency", "moneda"),
    "latitude": ("lat", "latitude"),
    "longitude": ("lon", "longitude", "lng"),
    "antiquity": ("antiquity", "years", "year_built", "construction_year"),
}


def _find_column(con: duckdb.DuckDBPyConnection, candidates: Iterable[str]) -> Optional[str]:
    columns = {
        row[1].lower(): row[1]
        for row in con.execute("PRAGMA table_info(raw)").fetchall()
    }
    lowered_to_original = {name.lower(): name for name in columns.values()}

    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in lowered_to_original:
            return lowered_to_original[lowered]
    for original in columns.values():
        for candidate in candidates:
            if candidate.lower() in original.lower():
                return original
    return None


def _ensure_required_columns(con: duckdb.DuckDBPyConnection) -> Dict[str, str]:
    resolved = {}
    for key, candidates in REQUIRED_FEATURES.items():
        column = _find_column(con, candidates)
        if column is None:
            raise ColumnNotFoundError(f"Required feature '{key}' with candidates {candidates} not found in dataset")
        resolved[key] = column
    return resolved


def _resolve_optional_columns(con: duckdb.DuckDBPyConnection) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for key, candidates in OPTIONAL_FEATURES.items():
        resolved[key] = _find_column(con, candidates)
    return resolved


def clean_dataset_with_duckdb(csv_path: str, output_path: str = str(SQL_DATA_PATH)) -> None:
    """Load the raw CSV using DuckDB, run SQL cleaning, and export parquet."""
    LOGGER.info("Cleaning data with DuckDB from %s", csv_path)
    con = duckdb.connect(database=":memory:")

    con.execute(
        "CREATE TABLE raw AS SELECT * FROM read_csv_auto(?, sample_size=-1, ignore_errors=true)",
        [csv_path],
    )

    required_cols = _ensure_required_columns(con)
    optional_cols = _resolve_optional_columns(con)

    price_col = required_cols["price"]
    surface_col = required_cols["surface_total"]

    select_clauses = [
        f"CAST({price_col} AS DOUBLE) AS target_price",
        f"CAST({surface_col} AS DOUBLE) AS surface_total_m2",
    ]

    def _add_optional(column_key: str, alias: str, cast: str = "DOUBLE") -> None:
        source = optional_cols.get(column_key)
        if source:
            select_clauses.append(f"CAST({source} AS {cast}) AS {alias}")
        else:
            select_clauses.append(f"CAST(NULL AS {cast}) AS {alias}")

    _add_optional("surface_covered", "surface_covered_m2")
    _add_optional("rooms", "rooms")
    _add_optional("bedrooms", "bedrooms")
    _add_optional("bathrooms", "bathrooms")
    _add_optional("latitude", "latitude")
    _add_optional("longitude", "longitude")
    _add_optional("antiquity", "antiquity_years")

    for text_key, alias in [
        ("property_type", "property_type"),
        ("state_name", "state_name"),
        ("neighborhood", "neighborhood"),
        ("currency", "currency"),
    ]:
        source = optional_cols.get(text_key)
        if source:
            select_clauses.append(f"CAST({source} AS VARCHAR) AS {alias}")
        else:
            select_clauses.append("CAST(NULL AS VARCHAR) AS {alias}".format(alias=alias))

    base_query = f"""
        SELECT
            {', '.join(select_clauses)}
        FROM raw
        WHERE {price_col} IS NOT NULL
          AND {surface_col} IS NOT NULL
          AND TRY_CAST({surface_col} AS DOUBLE) > 0
          AND TRY_CAST({price_col} AS DOUBLE) > 0
    """

    currency_col = optional_cols.get("currency")
    if currency_col:
        base_query += f"\n          AND {currency_col} = 'USD'"

    LOGGER.debug("Base SQL query:\n%s", base_query)

    con.execute(f"CREATE OR REPLACE TABLE base AS {base_query}")

    enrichment_query = """
        SELECT
            *,
            target_price / NULLIF(surface_total_m2, 0) AS price_per_m2,
            CASE WHEN surface_total_m2 > 0 THEN surface_covered_m2 / surface_total_m2 ELSE NULL END AS covered_ratio,
            CASE WHEN surface_total_m2 > 0 THEN rooms / surface_total_m2 ELSE NULL END AS rooms_per_surface,
            CASE WHEN antiquity_years IS NOT NULL AND antiquity_years > 0 THEN 2024 - antiquity_years ELSE NULL END AS property_age
        FROM base
    """

    con.execute("CREATE OR REPLACE TABLE enriched AS " + enrichment_query)

    LOGGER.info("Writing cleaned parquet to %s", output_path)
    con.execute(
        "COPY (SELECT * FROM enriched) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [output_path],
    )

    summary = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            AVG(target_price) AS avg_price,
            AVG(surface_total_m2) AS avg_surface_total,
            MIN(target_price) AS min_price,
            MAX(target_price) AS max_price
        FROM enriched
        """
    ).fetchone()

    DUCKDB_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "rows": summary[0],
                "avg_price": summary[1],
                "avg_surface_total": summary[2],
                "min_price": summary[3],
                "max_price": summary[4],
            },
            indent=2,
        )
    )

    con.close()


