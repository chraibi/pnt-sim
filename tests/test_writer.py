"""Lock down the sqlite schema.

The whole point of pnt-sim is that the Web-Based JuPedSim viewer reads its
output unchanged. If any of these column/table names drift, that contract
breaks silently. Mirrors what backend/models.py SqliteResults queries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pnt_sim.writer import DATABASE_VERSION, TrajectoryWriter

WKT_SQUARE = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"


@pytest.fixture
def trail(tmp_path: Path) -> Path:
    out = tmp_path / "trails.sqlite"
    with TrajectoryWriter(out, WKT_SQUARE, fps=10.0) as w:
        w.write_frame(0, [(1, 1.0, 1.0, 1.0, 0.0), (2, 2.0, 2.0, 0.0, 1.0)])
        w.write_frame(1, [(1, 1.1, 1.0, 1.0, 0.0), (2, 2.0, 2.1, 0.0, 1.0)])
    return out


def _table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def test_required_tables_exist(trail: Path) -> None:
    con = sqlite3.connect(trail)
    names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # v2 tables — the app's upload path (backend/routes/simulation.py) only
    # accepts versions 1 and 2; v3's `levels`/`landings` are intentionally absent.
    assert names >= {"trajectory_data", "metadata", "geometry", "frame_data"}
    assert "levels" not in names
    assert "landings" not in names


def test_trajectory_data_columns_match_jupedsim_v2(trail: Path) -> None:
    con = sqlite3.connect(trail)
    cols = _table_columns(con, "trajectory_data")
    assert cols == ["frame", "id", "pos_x", "pos_y", "ori_x", "ori_y"]


def test_metadata_has_version_fps_and_bounds(trail: Path) -> None:
    con = sqlite3.connect(trail)
    rows = dict(con.execute("SELECT key, value FROM metadata"))
    assert rows["version"] == str(DATABASE_VERSION)
    assert float(rows["fps"]) == 10.0
    for key in ("xmin", "xmax", "ymin", "ymax"):
        assert key in rows, f"missing metadata key {key}"
    assert float(rows["xmin"]) == 0.0
    assert float(rows["xmax"]) == 10.0


def test_geometry_and_frame_data_link(trail: Path) -> None:
    con = sqlite3.connect(trail)
    geos = list(con.execute("SELECT hash, wkt FROM geometry"))
    assert len(geos) == 1
    geo_hash, wkt = geos[0]
    assert wkt == WKT_SQUARE
    frames = list(con.execute("SELECT frame, geometry_hash FROM frame_data ORDER BY frame"))
    assert frames == [(0, geo_hash), (1, geo_hash)]


def test_trajectory_data_rows_match_what_was_written(trail: Path) -> None:
    con = sqlite3.connect(trail)
    rows = list(
        con.execute("SELECT frame, id, pos_x, pos_y, ori_x, ori_y FROM trajectory_data ORDER BY frame, id")
    )
    assert rows == [
        (0, 1, 1.0, 1.0, 1.0, 0.0),
        (0, 2, 2.0, 2.0, 0.0, 1.0),
        (1, 1, 1.1, 1.0, 1.0, 0.0),
        (1, 2, 2.0, 2.1, 0.0, 1.0),
    ]
