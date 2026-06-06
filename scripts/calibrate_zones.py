"""
Interactive zone calibration tool.

Loads the saved calibration frame and lets you click corners for each zone.
Press any key to advance to the next zone. Writes results to config/config.yaml.

Usage:
    python scripts/calibrate_zones.py
"""

import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
FRAME_PATH = ROOT / "tests" / "data" / "wide-angle" / "calibration_frame.jpg"

ZONES = ["entry", "sink", "exit"]


def collect_zone(img, zone_name: str) -> list:
    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"  {zone_name} point {len(points)}: ({x}, {y})")

    print(f"\nZone: {zone_name.upper()} — click corners, then press any key when done.")
    cv2.imshow("Calibration", img)
    cv2.setMouseCallback("Calibration", click)
    cv2.waitKey(0)
    return points


def update_config(polygons: dict) -> None:
    text = CONFIG_PATH.read_text()
    for zone, pts in polygons.items():
        pts_str = str(pts).replace(" ", "")
        pattern = rf"(  {zone}:\n    polygon:).*"
        text, n = re.subn(pattern, rf"\1 {pts_str}", text)
        if n == 0:
            print(f"WARNING: zone '{zone}' not found in config.yaml")
    CONFIG_PATH.write_text(text)


def main():
    if not FRAME_PATH.exists():
        sys.exit(f"ERROR: calibration frame not found at {FRAME_PATH}")

    img = cv2.imread(str(FRAME_PATH))
    if img is None:
        sys.exit(f"ERROR: could not load {FRAME_PATH}")

    print(f"Loaded: {FRAME_PATH}")
    print("For each zone: click corners, press any key to move on.")

    polygons = {}
    for zone in ZONES:
        polygons[zone] = collect_zone(img, zone)
        print(f"  saved {zone}: {polygons[zone]}")

    cv2.destroyAllWindows()
    update_config(polygons)

    print("\nconfig.yaml updated. Final zones:")
    for zone, pts in polygons.items():
        print(f"  {zone}: {pts}")


if __name__ == "__main__":
    main()
