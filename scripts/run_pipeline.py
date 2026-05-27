"""
Entry-point: run the dual-camera handwash compliance pipeline.

Usage:
    # Use sources defined in config/config.yaml
    python scripts/run_pipeline.py

    # Override both camera sources on the command line
    python scripts/run_pipeline.py --source 0 --source-topdown 2

    # Use video files for testing (no real cameras needed)
    python scripts/run_pipeline.py --source tests/data/wide.mp4 --source-topdown tests/data/topdown.mp4

    # Override config file location
    python scripts/run_pipeline.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handwash.pipeline import HandwashPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_source(value: str | None):
    """Convert a source string to int (camera index) if numeric, else keep as string path/URL."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Handwash compliance pipeline — dual-camera edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    p.add_argument(
        "--source",
        default=None,
        metavar="WIDE",
        help="Wide-angle camera: index (e.g. 0) or path/RTSP URL. "
             "Overrides cameras.wide.source in config.",
    )
    p.add_argument(
        "--source-topdown",
        default=None,
        metavar="TOPDOWN",
        help="Top-down camera: index (e.g. 2) or path/RTSP URL. "
             "Overrides cameras.topdown.source in config.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = HandwashPipeline(config_path=args.config)
    pipeline.run(
        source_wide=_parse_source(args.source),
        source_topdown=_parse_source(args.source_topdown),
    )
