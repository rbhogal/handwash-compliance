"""Tests for ZoneEngine polygon membership."""

import numpy as np
import pytest

from handwash.compliance.zone_engine import ZoneEngine


ZONES = {
    "entry": [[0, 0], [1280, 0], [1280, 200], [0, 200]],
    "sink":  [[400, 200], [880, 200], [880, 600], [400, 600]],
    "exit":  [[0, 520], [1280, 520], [1280, 720], [0, 720]],
}


@pytest.fixture
def engine():
    return ZoneEngine(ZONES)


def test_centroid_in_sink(engine):
    result = engine.query((640, 400))   # centre of sink zone
    assert result["sink"] is True
    assert result["entry"] is False
    assert result["exit"] is False


def test_centroid_in_entry(engine):
    result = engine.query((640, 100))
    assert result["entry"] is True
    assert result["sink"] is False


def test_centroid_in_exit(engine):
    result = engine.query((640, 600))
    assert result["exit"] is True
    assert result["sink"] is False


def test_centroid_in_corner(engine):
    result = engine.query((0, 0))
    assert result["entry"] is True


def test_centroid_outside_all(engine):
    result = engine.query((200, 400))   # between entry and sink, left of sink
    assert not any(result.values())
