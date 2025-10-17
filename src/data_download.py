"""Utilities to download the Kaggle dataset locally."""
from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from .paths import RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

DATASET_SLUG = "martinbasualdo/property-prices-in-caba-zonaprop-data"


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        for child in src.iterdir():
            target = dst / child.name
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                _copy_tree(child, target)
            else:
                shutil.copy2(child, target)
    else:
        shutil.copy2(src, dst)


def _extract_archives(folder: Path) -> None:
    for archive in folder.glob("*.zip"):
        LOGGER.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(folder)


def download_dataset(raw_dir: Optional[Path] = None) -> Path:
    """Download the Kaggle dataset and return the path to the main CSV file."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ModuleNotFoundError(
            "kagglehub is required to download the dataset. Install it with `pip install kagglehub`."
        ) from exc

    raw_dir = raw_dir or RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading dataset %s", DATASET_SLUG)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    LOGGER.info("Dataset downloaded to %s", dataset_path)

    temp_target = raw_dir / dataset_path.name
    if temp_target.exists():
        shutil.rmtree(temp_target)
    temp_target.mkdir(parents=True, exist_ok=True)

    _copy_tree(dataset_path, temp_target)
    _extract_archives(temp_target)

    csv_files = list(temp_target.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files were found in the downloaded dataset. Please check the Kaggle dataset structure."
        )
    # Pick the largest CSV assuming it contains the listings.
    main_csv = max(csv_files, key=lambda path: path.stat().st_size)
    LOGGER.info("Selected %s as main dataset file", main_csv)
    return main_csv

