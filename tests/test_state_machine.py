"""Tests for the compliance state machine."""

import time
import pytest

from handwash.compliance.state_machine import ComplianceStateMachine, WashState


def make_sm(**kwargs):
    defaults = dict(min_wash_duration_sec=20.0, max_entry_to_sink_sec=10.0)
    defaults.update(kwargs)
    return ComplianceStateMachine(**defaults)


class TestStateMachine:
    def test_initial_state_on_entry(self):
        sm = make_sm()
        sm.update(1, in_entry_zone=True, in_sink_zone=False, in_exit_zone=False)
        assert 1 in sm._records
        assert sm._records[1].state == WashState.ENTERED

    def test_transition_to_at_sink(self):
        sm = make_sm()
        sm.update(1, in_entry_zone=True,  in_sink_zone=False, in_exit_zone=False)
        sm.update(1, in_entry_zone=False, in_sink_zone=True,  in_exit_zone=False)
        assert sm._records[1].state == WashState.AT_SINK

    def test_violation_on_skip(self):
        sm = make_sm()
        sm.update(1, in_entry_zone=True,  in_sink_zone=False, in_exit_zone=False)
        sm.update(1, in_entry_zone=False, in_sink_zone=False, in_exit_zone=True)
        verdicts = sm.pop_verdicts()
        assert len(verdicts) == 1
        assert verdicts[0].compliant is False
        assert "no sink" in verdicts[0].message.lower()

    def test_step_collection_in_washing_state(self):
        sm = make_sm()
        sm.update(1, in_entry_zone=True,  in_sink_zone=False, in_exit_zone=False)
        sm.update(1, in_entry_zone=False, in_sink_zone=True,  in_exit_zone=False)
        # Force into WASHING state
        sm._records[1].state = WashState.WASHING
        sm.update(1, in_entry_zone=False, in_sink_zone=True, in_exit_zone=False, gesture_step_id=3)
        assert 3 in sm._records[1].steps_detected

    def test_track_lost_generates_verdict(self):
        sm = make_sm()
        sm.update(1, in_entry_zone=True, in_sink_zone=False, in_exit_zone=False)
        sm.track_lost(1)
        verdicts = sm.pop_verdicts()
        assert len(verdicts) == 1
