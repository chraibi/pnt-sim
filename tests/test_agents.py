"""Tests for the Penn & Turner agent loop and visibility math."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from pnt_sim.agents import AgentParams, AgentSimulation
from pnt_sim.visibility import VisibilityGrid, build_visibility


def _open_square_grid(size: float = 10.0, spacing: float = 1.0) -> VisibilityGrid:
    return build_visibility(
        Polygon([(0, 0), (size, 0), (size, size), (0, size)]),
        spacing,
        min_wall_distance=0.05,
    )


def test_fov_filter_drops_cells_behind_the_agent() -> None:
    grid = _open_square_grid()
    params = AgentParams(fov_deg=170.0, steps_before_turn=3, step_length=1.0)
    rng = np.random.default_rng(0)
    sim = AgentSimulation(grid, params, rng)

    aid = sim.spawn(5.0, 5.0)
    # Force a known heading: due east (0 rad). Anything past heading±85° is dropped.
    agent = sim._agents[0]  # noqa: SLF001 — white-box test
    agent.heading = 0.0
    cell_idx = sim._kdtree.query([agent.x, agent.y])[1]
    visible = grid.visible[cell_idx]
    cone = sim._fov_filter(agent, visible)  # noqa: SLF001
    assert cone.size > 0
    # Every kept cell must lie ahead within ±85°.
    deltas = []
    for idx in cone:
        tx, ty = grid.cells[idx]
        angle = math.atan2(ty - agent.y, tx - agent.x)
        d = (angle - agent.heading + math.pi) % (2 * math.pi) - math.pi
        deltas.append(abs(d))
    assert all(d <= math.radians(85) + 1e-9 for d in deltas)
    assert aid == 1


def test_agent_moves_one_step_per_frame() -> None:
    grid = _open_square_grid()
    params = AgentParams(fov_deg=170.0, steps_before_turn=3, step_length=1.0)
    sim = AgentSimulation(grid, params, np.random.default_rng(1))
    sim.spawn(5.0, 5.0)
    p0 = (sim._agents[0].x, sim._agents[0].y)  # noqa: SLF001
    sim.step()
    p1 = (sim._agents[0].x, sim._agents[0].y)  # noqa: SLF001
    assert math.isclose(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 1.0, abs_tol=1e-9)


def test_agent_disappears_after_lifetime() -> None:
    grid = _open_square_grid()
    params = AgentParams(steps_before_turn=3, step_length=1.0, agent_lifetime=4)
    sim = AgentSimulation(grid, params, np.random.default_rng(2))
    sim.spawn(5.0, 5.0)
    counts = []
    for _ in range(6):
        sim.step()
        counts.append(sim.alive_count)
    # Lifetime 4 -> alive for frames 0,1,2,3 then removed on frame 4.
    assert counts == [1, 1, 1, 0, 0, 0]


def test_visibility_in_open_room_is_dense() -> None:
    grid = _open_square_grid(size=5.0, spacing=1.0)
    # In a convex room with no obstacles, every cell sees every other cell.
    n = len(grid.cells)
    avg = sum(v.size for v in grid.visible) / n
    assert avg == pytest.approx(n - 1)


def test_visibility_blocked_by_interior_wall() -> None:
    # Two rooms separated by a wall with a doorway gap.
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    # Hole = obstacle: a thin vertical wall from y=1 to y=9 at x=5.
    wall = [(4.9, 1), (5.1, 1), (5.1, 9), (4.9, 9)]
    walkable = Polygon(outer, holes=[wall])
    grid = build_visibility(walkable, 1.0, min_wall_distance=0.05)

    # Pick a cell on each side of the wall.
    cells = grid.cells
    left = np.argmin(np.hypot(cells[:, 0] - 2.0, cells[:, 1] - 5.0))
    right = np.argmin(np.hypot(cells[:, 0] - 8.0, cells[:, 1] - 5.0))
    assert right not in grid.visible[left]
    assert left not in grid.visible[right]


def test_scenario_io_parses_minimal_app_config(tmp_path) -> None:
    import json
    from pnt_sim.scenario_io import load_scenario

    cfg = {
        "exits": {},
        "distributions": {
            "jps-distributions_0": {
                "type": "polygon",
                "coordinates": [[1, 1], [3, 1], [3, 3], [1, 3]],
            }
        },
    }
    wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    json_path = tmp_path / "config.json"
    wkt_path = tmp_path / "geometry.wkt"
    json_path.write_text(json.dumps(cfg))
    wkt_path.write_text(wkt)
    scenario = load_scenario(json_path, wkt_path)
    assert scenario.walkable_wkt == wkt
    assert scenario.walkable.bounds == (0.0, 0.0, 10.0, 10.0)
    assert len(scenario.release_zones) == 1
    assert scenario.release_zones[0].name == "jps-distributions_0"
