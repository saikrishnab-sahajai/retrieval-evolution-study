"""
Download all BEIR datasets used in this study.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets scifact fiqa trec-covid
    python scripts/download_datasets.py --list

Data is written to data/datasets/<dataset_name>/.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "datasets.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def download_beir_dataset(name: str, data_dir: Path) -> Path:
    """Download a BEIR dataset and return the local path."""
    from beir import util as beir_util

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    out_dir = data_dir / name

    if out_dir.exists():
        logger.info(f"  {name}: already downloaded at {out_dir}")
        return out_dir

    logger.info(f"  Downloading {name} from BEIR...")
    data_path = beir_util.download_and_unzip(url, str(data_dir))
    logger.info(f"  {name}: saved to {data_path}")
    return Path(data_path)


def main():
    parser = argparse.ArgumentParser(description="Download BEIR datasets")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to download (default: all enabled in datasets.yaml)"
    )
    parser.add_argument("--list", action="store_true", help="List configured datasets and exit")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = REPO_ROOT / cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = cfg["datasets"]

    if args.list:
        print("\nConfigured datasets:")
        for k, v in all_datasets.items():
            status = "enabled" if v["enabled"] else "disabled"
            print(f"  {k:<15} {v['passages']:>12,} passages   [{status}]")
            print(f"             {v['description']}")
        return

    # Determine which datasets to download
    if args.datasets:
        to_download = {k: v for k, v in all_datasets.items() if k in args.datasets}
        missing = set(args.datasets) - set(to_download.keys())
        if missing:
            logger.error(f"Unknown dataset(s): {missing}. Use --list to see options.")
            sys.exit(1)
    else:
        to_download = {k: v for k, v in all_datasets.items() if v["enabled"]}

    logger.info(f"Downloading {len(to_download)} dataset(s) to {data_dir}...")
    for name, meta in to_download.items():
        beir_name = meta["beir_name"]
        try:
            download_beir_dataset(beir_name, data_dir)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
