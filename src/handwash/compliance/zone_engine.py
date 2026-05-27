"""
Zone Rule Engine
Determines whether a point (person centroid) lies inside a named polygon zone.
Zones are defined in config/config.yaml and loaded once per camera installation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]   # (x, y)
Polygon = np.ndarray          # shape (N, 2), dtype float32


class ZoneEngine:
    """
    Loads polygon zone definitions and tests centroid membership.

    Usage:
        zones = {
            "entry": [[0,0],[1280,0],[1280,200],[0,200]],
            "sink":  [[400,200],[880,200],[880,600],[400,600]],
            "exit":  [[0,520],[1280,520],[1280,720],[0,720]],
        }
        engine = ZoneEngine(zones)
        memberships = engine.query((cx, cy))
        # {'entry': False, 'sink': True, 'exit': False}
    """

    def __init__(self, zones: Dict[str, List[List[int]]]) -> None:
        self._polygons: Dict[str, Polygon] = {}
        for name, vertices in zones.items():
            self._polygons[name] = np.array(vertices, dtype=np.float32)

    def query(self, centroid: Point) -> Dict[str, bool]:
        """
        Test centroid membership for every zone.

        Args:
            centroid: (x, y) in pixel coordinates.

        Returns:
            Dict of {zone_name: bool}
        """
        pt = (float(centroid[0]), float(centroid[1]))
        return {
            name: self._point_in_polygon(pt, poly)
            for name, poly in self._polygons.items()
        }

    @staticmethod
    def _point_in_polygon(point: Point, polygon: Polygon) -> bool:
        """Use OpenCV's pointPolygonTest (1.0 = inside, -1.0 = outside)."""
        result = cv2.pointPolygonTest(polygon, point, measureDist=False)
        return result >= 0

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Overlay zone polygons on a frame (for debugging / dashboard)."""
        colours = {
            "entry": (255, 200, 0),    # blue-ish
            "sink":  (0, 200, 255),    # yellow
            "exit":  (50, 50, 255),    # red
        }
        overlay = frame.copy()
        for name, poly in self._polygons.items():
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            colour = colours.get(name, (200, 200, 200))
            cv2.polylines(overlay, [pts], isClosed=True, color=colour, thickness=2)
            # Label
            centroid = poly.mean(axis=0).astype(int)
            cv2.putText(
                overlay, name, tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
