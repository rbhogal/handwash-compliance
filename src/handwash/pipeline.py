"""
Main pipeline orchestrator — dual-camera edition.
Wires together all four layers: Detection → Tracking → Zone Engine → Gesture → Compliance.

Camera roles
------------
- Wide-angle (cameras.wide):  zone tracking (entry / sink / exit) + person IDs.
- Top-down   (cameras.topdown): gesture classification only — full frame or
  configured ROI is fed directly to the gesture classifier whenever a person
  is confirmed in the sink zone by the wide-angle tracker.

Run via:
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --source 0 --source-topdown 2
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .detection import HandwashDetector
from .tracking import PersonTracker
from .gesture import GestureClassifier
from .compliance import (
    ComplianceStateMachine,
    ZoneEngine,
    LEDController,
    create_app,
)
from .compliance.dashboard import record_event

logger = logging.getLogger(__name__)


class HandwashPipeline:
    """
    End-to-end dual-camera pipeline.

    Wide-angle camera → zone tracking, person IDs.
    Top-down camera   → gesture classification (full frame or ROI crop).

    Args:
        config_path: Path to config/config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        cfg = self._load_config(config_path)

        # ── Camera configs ────────────────────────────────────────
        cam_cfg = cfg.get("cameras", {})
        self._wide_cfg = cam_cfg.get("wide", {})
        self._topdown_cfg = cam_cfg.get("topdown", {})

        # ── Layer 1: Detection ────────────────────────────────────
        # NOTE: HandwashDetector is instantiated but not called in the live
        # pipeline loop. Zone membership uses fixed config.yaml polygons;
        # person tracking uses PersonTracker (ByteTrack) below.
        # Reserved for future use: automatic zone calibration, multi-sink
        # environments, or soap dispenser confirmation logic.
        self.detector = HandwashDetector(
            weights=cfg["detection"]["weights"],
            conf_threshold=cfg["detection"]["conf_threshold"],
            iou_threshold=cfg["detection"]["iou_threshold"],
            input_size=cfg["detection"]["input_size"],
        )

        # ── Layer 1: Tracking (wide-angle only) ───────────────────
        self.tracker = PersonTracker(
            weights=cfg["detection"]["weights"],
            conf_threshold=cfg["detection"]["conf_threshold"],
        )

        # ── Zone Engine (wide-angle pixel coords) ─────────────────
        self.zone_engine = ZoneEngine(cfg["zones"])

        # ── Layer 2: Gesture Classifier (top-down frames) ─────────
        gesture_cfg = cfg["gesture"]
        self._gesture_roi: Optional[list] = self._topdown_cfg.get("gesture_roi")
        self._gesture_input_size: int = gesture_cfg["input_size"]
        self.gesture_classifier = GestureClassifier(
            weights=gesture_cfg["weights"],
            sequence_length=gesture_cfg["sequence_length"],
            conf_threshold=gesture_cfg["conf_threshold"],
            input_size=self._gesture_input_size,
        )

        # ── Compliance & Output ───────────────────────────────────
        compliance_cfg = cfg["compliance"]
        self.state_machine = ComplianceStateMachine(
            min_wash_duration_sec=compliance_cfg["min_wash_duration_sec"],
            max_entry_to_sink_sec=compliance_cfg["max_entry_to_sink_sec"],
        )
        self.led = LEDController(gpio_pin=cfg["output"]["led_gpio_pin"])

        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(
        self,
        source_wide: Optional[str | int] = None,
        source_topdown: Optional[str | int] = None,
    ) -> None:
        """
        Open both camera sources and process frames in a loop.
        Press 'q' to quit.

        Args:
            source_wide:    Wide-angle camera index / path / RTSP URL.
                            Falls back to cameras.wide.source in config.
            source_topdown: Top-down camera index / path / RTSP URL.
                            Falls back to cameras.topdown.source in config.
        """
        src_wide = source_wide if source_wide is not None else self._wide_cfg.get("source", 0)
        src_topdown = source_topdown if source_topdown is not None else self._topdown_cfg.get("source", 2)

        cap_wide = cv2.VideoCapture(src_wide)
        cap_topdown = cv2.VideoCapture(src_topdown)

        if not cap_wide.isOpened():
            raise RuntimeError(f"Cannot open wide-angle camera source: {src_wide}")
        if not cap_topdown.isOpened():
            raise RuntimeError(f"Cannot open top-down camera source: {src_topdown}")

        # Optional: hint preferred resolution / fps to the driver
        for cap, cfg_key in [(cap_wide, "wide"), (cap_topdown, "topdown")]:
            cam = self._cfg.get("cameras", {}).get(cfg_key, {})
            if cam.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            if cam.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            if cam.get("fps"):
                cap.set(cv2.CAP_PROP_FPS, cam["fps"])

        # Start dashboard in background thread
        dash_thread = threading.Thread(
            target=lambda: create_app().run(
                host="0.0.0.0",
                port=self._cfg["output"]["dashboard_port"],
                use_reloader=False,
                debug=False,
            ),
            daemon=True,
        )
        dash_thread.start()
        logger.info(
            f"Dashboard running at http://localhost:{self._cfg['output']['dashboard_port']}"
        )

        prev_track_ids: set[int] = set()

        # Keep last good frames so we can degrade gracefully if one camera drops
        last_frame_wide: Optional[np.ndarray] = None
        last_frame_topdown: Optional[np.ndarray] = None

        try:
            while True:
                ret_w, frame_w = cap_wide.read()
                ret_t, frame_t = cap_topdown.read()

                if not ret_w:
                    if last_frame_wide is None:
                        logger.error("Wide-angle camera failed on first frame — exiting.")
                        break
                    logger.warning("Wide-angle camera dropped a frame — using last good frame.")
                    frame_w = last_frame_wide
                else:
                    last_frame_wide = frame_w

                if not ret_t:
                    if last_frame_topdown is None:
                        logger.error("Top-down camera failed on first frame — exiting.")
                        break
                    logger.warning("Top-down camera dropped a frame — using last good frame.")
                    frame_t = last_frame_topdown
                else:
                    last_frame_topdown = frame_t

                combined = self._process_frame(frame_w, frame_t, prev_track_ids)

                cv2.imshow("Handwash Compliance", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap_wide.release()
            cap_topdown.release()
            cv2.destroyAllWindows()
            self.led.cleanup()

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame_wide: np.ndarray,
        frame_topdown: np.ndarray,
        prev_track_ids: set,
    ) -> np.ndarray:
        """
        Process one pair of frames and return the side-by-side visualisation.

        Wide-angle  → tracking, zone membership, state machine.
        Top-down    → gesture classification crop (when person is at sink).
        """
        # 1. Track persons on the wide-angle frame
        tracks = self.tracker.update(frame_wide)
        current_ids = set(tracks.keys())

        # Notify state machine of lost tracks
        for lost_id in prev_track_ids - current_ids:
            self.state_machine.track_lost(lost_id)
        prev_track_ids.clear()
        prev_track_ids.update(current_ids)

        # 2. Per-person zone + gesture update
        last_gesture_pred = None
        for tid, track in tracks.items():
            zones = self.zone_engine.query(track.centroid)

            # Gesture classification: use top-down camera crop
            gesture_step = None
            if zones.get("sink"):
                roi = self._get_topdown_roi(frame_topdown)
                self.gesture_classifier.push_frame(roi)
                if self.gesture_classifier.ready():
                    pred = self.gesture_classifier.classify()
                    if pred is not None:
                        gesture_step = pred.step_id
                        last_gesture_pred = pred

            self.state_machine.update(
                track_id=tid,
                in_entry_zone=zones.get("entry", False),
                in_sink_zone=zones.get("sink", False),
                in_exit_zone=zones.get("exit", False),
                gesture_step_id=gesture_step,
            )

        # 3. Process compliance verdicts
        for verdict in self.state_machine.pop_verdicts():
            logger.info(f"[Track {verdict.track_id}] {verdict.message}")
            self.led.signal_verdict(verdict.compliant)
            record_event(
                track_id=verdict.track_id,
                compliant=verdict.compliant,
                steps_detected=list(verdict.steps_detected),
                wash_duration_sec=verdict.wash_duration_sec,
                message=verdict.message,
            )

        # 4. Build visualisation — wide-angle with overlays + top-down with gesture label
        vis_wide = self.zone_engine.draw_zones(frame_wide.copy())
        for tid, track in tracks.items():
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(vis_wide, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_wide, f"ID:{tid}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        return self._make_side_by_side(vis_wide, frame_topdown, last_gesture_pred)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_topdown_roi(self, frame_topdown: np.ndarray) -> np.ndarray:
        """
        Extract the gesture region from the top-down camera frame.

        If ``cameras.topdown.gesture_roi`` is set (``[x1, y1, x2, y2]``),
        return that crop; otherwise return the full frame.
        The result is NOT resized here — GestureClassifier handles that.
        """
        if self._gesture_roi is not None:
            x1, y1, x2, y2 = [int(v) for v in self._gesture_roi]
            h, w = frame_topdown.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_topdown[y1:y2, x1:x2]
            if crop.size > 0:
                logger.debug(
                    f"Top-down gesture ROI crop: ({x1},{y1})→({x2},{y2})"
                )
                return crop
        logger.debug("Top-down gesture: using full frame")
        return frame_topdown

    def _make_side_by_side(
        self,
        vis_wide: np.ndarray,
        frame_topdown: np.ndarray,
        gesture_pred,
    ) -> np.ndarray:
        """
        Combine the wide-angle frame (with overlays) and the top-down frame
        into a single side-by-side display image.

        The top-down panel is scaled to the same height as the wide-angle panel
        and labelled with the current gesture prediction.
        """
        h_wide = vis_wide.shape[0]

        # Resize top-down to match wide-angle height, preserving aspect ratio
        h_td, w_td = frame_topdown.shape[:2]
        scale = h_wide / h_td
        new_w = int(w_td * scale)
        vis_topdown = cv2.resize(frame_topdown, (new_w, h_wide))

        # Draw gesture label on top-down panel
        if gesture_pred is not None:
            label = f"Step {gesture_pred.step_id + 1}/7: {gesture_pred.step_name}"
            conf_pct = int(gesture_pred.confidence * 100)
            label_conf = f"{label} ({conf_pct}%)"
            cv2.putText(
                vis_topdown, label_conf, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        else:
            cv2.putText(
                vis_topdown, "Gesture: waiting...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2,
            )

        # Draw camera labels
        cv2.putText(
            vis_wide, "WIDE (zone tracking)", (10, vis_wide.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.putText(
            vis_topdown, "TOP-DOWN (gesture)", (10, h_wide - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        return np.hstack([vis_wide, vis_topdown])

    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
