"""
Per-person compliance state machine.

State transitions:
    ENTERED → AT_SINK → WASHING → EXITED  (compliant)
    ENTERED → EXITED (skipped sink)        (violation)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class WashState(Enum):
    ENTERED  = auto()   # Person detected in entry zone
    AT_SINK  = auto()   # Person centroid inside sink zone
    WASHING  = auto()   # Dwell time threshold met; gesture classifier active
    EXITED   = auto()   # Person left all zones — verdict is final
    VIOLATED = auto()   # Left without washing


@dataclass
class ComplianceVerdict:
    track_id: int
    compliant: bool
    steps_detected: Set[int]         # which WHO step IDs were observed
    wash_duration_sec: float
    message: str


@dataclass
class _PersonRecord:
    track_id: int
    state: WashState = WashState.ENTERED
    enter_time: float = field(default_factory=time.monotonic)
    sink_enter_time: Optional[float] = None
    wash_start_time: Optional[float] = None
    steps_detected: Set[int] = field(default_factory=set)

    # derived
    @property
    def time_in_sink(self) -> float:
        if self.sink_enter_time is None:
            return 0.0
        return time.monotonic() - self.sink_enter_time

    @property
    def wash_duration(self) -> float:
        if self.wash_start_time is None:
            return 0.0
        return time.monotonic() - self.wash_start_time


class ComplianceStateMachine:
    """
    Manages one _PersonRecord per active track.

    Call `update()` each frame with zone memberships and detected gesture steps.
    Retrieve final verdicts via `pop_verdicts()`.

    Args:
        min_wash_duration_sec: Minimum dwell time (WHO = 20s).
        max_entry_to_sink_sec: Window allowed between entry and reaching sink.
        required_steps:        Set of step IDs required for full compliance.
    """

    def __init__(
        self,
        min_wash_duration_sec: float = 20.0,
        max_entry_to_sink_sec: float = 10.0,
        required_steps: Optional[Set[int]] = None,
    ) -> None:
        self.min_wash_duration_sec = min_wash_duration_sec
        self.max_entry_to_sink_sec = max_entry_to_sink_sec
        self.required_steps = required_steps or set(range(6))

        self._records: Dict[int, _PersonRecord] = {}
        self._verdicts: List[ComplianceVerdict] = []

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        in_entry_zone: bool,
        in_sink_zone: bool,
        in_exit_zone: bool,
        gesture_step_id: Optional[int] = None,
    ) -> None:
        """
        Drive the state machine for one person for the current frame.

        Args:
            track_id:        ByteTrack ID.
            in_entry_zone:   Centroid inside entry polygon.
            in_sink_zone:    Centroid inside sink polygon.
            in_exit_zone:    Centroid inside exit polygon.
            gesture_step_id: WHO step just classified (None if classifier not active).
        """
        # Register new tracks
        if track_id not in self._records:
            self._records[track_id] = _PersonRecord(track_id=track_id)

        rec = self._records[track_id]

        # ── Collect gesture evidence ───────────────────────────────────
        if gesture_step_id is not None and rec.state == WashState.WASHING:
            rec.steps_detected.add(gesture_step_id)

        # ── State transitions ──────────────────────────────────────────
        if rec.state == WashState.ENTERED:
            if in_sink_zone:
                rec.state = WashState.AT_SINK
                rec.sink_enter_time = time.monotonic()
            elif in_exit_zone:
                # Left without visiting sink
                self._finalise(rec, skipped_sink=True)

        elif rec.state == WashState.AT_SINK:
            if in_sink_zone and rec.time_in_sink >= 1.0:
                # Enough dwell to start scoring
                rec.state = WashState.WASHING
                rec.wash_start_time = time.monotonic()
            elif not in_sink_zone:
                if in_exit_zone:
                    self._finalise(rec, skipped_sink=False)
                else:
                    # Stepped away briefly — stay AT_SINK
                    pass

        elif rec.state == WashState.WASHING:
            if rec.steps_detected >= self.required_steps:
                # All steps detected while still at sink — fire green immediately
                duration = rec.wash_duration
                self._verdicts.append(ComplianceVerdict(
                    track_id=rec.track_id,
                    compliant=True,
                    steps_detected=set(rec.steps_detected),
                    wash_duration_sec=duration,
                    message=f"Compliant: all 6 steps, {duration:.1f}s",
                ))
                del self._records[rec.track_id]
            elif not in_sink_zone:
                self._finalise(rec, skipped_sink=False)

    def track_lost(self, track_id: int) -> None:
        """Call when ByteTrack drops a track (person left camera FOV)."""
        if track_id in self._records:
            rec = self._records[track_id]
            if rec.state not in (WashState.EXITED, WashState.VIOLATED):
                self._finalise(rec, skipped_sink=(rec.state == WashState.ENTERED))

    # ------------------------------------------------------------------
    # Verdict collection
    # ------------------------------------------------------------------

    def pop_verdicts(self) -> List[ComplianceVerdict]:
        """Return and clear all pending verdicts."""
        out = list(self._verdicts)
        self._verdicts.clear()
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finalise(self, rec: _PersonRecord, skipped_sink: bool) -> None:
        if skipped_sink or rec.state == WashState.ENTERED:
            rec.state = WashState.VIOLATED
            verdict = ComplianceVerdict(
                track_id=rec.track_id,
                compliant=False,
                steps_detected=set(),
                wash_duration_sec=0.0,
                message="Violation: no sink interaction",
            )
        else:
            duration = rec.wash_duration
            missing_steps = self.required_steps - rec.steps_detected
            full_duration_ok = duration >= self.min_wash_duration_sec
            all_steps_ok = len(missing_steps) == 0

            compliant = full_duration_ok and all_steps_ok
            if compliant:
                msg = f"Compliant: all 6 steps, {duration:.1f}s"
            elif not full_duration_ok and not all_steps_ok:
                msg = f"Partial: {len(rec.steps_detected)}/6 steps, only {duration:.1f}s"
            elif not full_duration_ok:
                msg = f"Partial: duration too short ({duration:.1f}s < {self.min_wash_duration_sec}s)"
            else:
                msg = f"Partial: missed steps {missing_steps}"

            rec.state = WashState.EXITED
            verdict = ComplianceVerdict(
                track_id=rec.track_id,
                compliant=compliant,
                steps_detected=set(rec.steps_detected),
                wash_duration_sec=duration,
                message=msg,
            )

        self._verdicts.append(verdict)
        del self._records[rec.track_id]
