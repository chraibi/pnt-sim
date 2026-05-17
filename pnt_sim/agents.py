"""Penn & Turner EVA agent loop.

Each agent:
  - position (x, y) continuous
  - heading (radians)
  - target (x, y) — re-picked every K steps or on arrival
  - lifetime — frames remaining before the agent is removed

Per frame, every agent advances `step_length` along its heading. Every K
steps (or on arrival at the current target), the agent re-decides:

  1. Find the nearest grid cell to its position.
  2. Take that cell's visible-set (precomputed).
  3. Filter to a ±FOV/2 cone around the heading.
  4. Uniformly sample one of those cells as the new target. (If the cone
     is empty — e.g. agent facing a wall — relax to the full visible-set so
     the agent doesn't freeze.)
  5. Update heading to the direction of the new target.

Agents pass through each other. There's no collision model — Turner's
EVA agents represent independent visitation samples, not crowd dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Polygon

from .visibility import VisibilityGrid


@dataclass
class AgentParams:
    fov_deg: float = 170.0
    steps_before_turn: int = 3
    step_length: float | None = None  # default = grid spacing
    agent_lifetime: int = 1000


@dataclass
class _Agent:
    aid: int
    x: float
    y: float
    heading: float  # radians
    target_idx: int  # cell index of current target
    steps_until_decision: int
    lifetime: int


class AgentSimulation:
    """Drive a population of EVA agents over a precomputed visibility grid."""

    def __init__(
        self,
        grid: VisibilityGrid,
        params: AgentParams,
        rng: np.random.Generator,
    ) -> None:
        self._grid = grid
        self._params = params
        self._rng = rng
        self._step_length = params.step_length or grid.grid_spacing
        self._half_fov = math.radians(params.fov_deg) / 2.0
        self._kdtree = cKDTree(grid.cells)
        self._walls = grid.walls
        self._wall_tree = STRtree(grid.walls) if grid.walls else None
        self._agents: list[_Agent] = []
        self._next_id = 1

    @property
    def alive_count(self) -> int:
        return len(self._agents)

    def spawn(self, x: float, y: float) -> int:
        """Spawn one agent at (x, y).

        Per T&P Figure 8, the initial destination is picked from the full
        visible set (omni), *not* through the FOV cone — the agent has no
        prior heading to constrain the cone against. Subsequent re-picks
        (every n steps) use the FOV cone.
        """
        cell_idx = int(self._kdtree.query([x, y])[1])
        visible = self._grid.visible[cell_idx]
        if visible.size > 0:
            target_idx = int(self._rng.choice(visible))
            target_xy = self._grid.cells[target_idx]
            heading = math.atan2(target_xy[1] - y, target_xy[0] - x)
        else:
            # Boxed in — degenerate but defined. Random heading, self-target.
            target_idx = cell_idx
            heading = float(self._rng.uniform(0.0, 2.0 * math.pi))
        agent = _Agent(
            aid=self._next_id,
            x=float(x),
            y=float(y),
            heading=heading,
            target_idx=target_idx,
            steps_until_decision=self._params.steps_before_turn,
            lifetime=self._params.agent_lifetime,
        )
        self._next_id += 1
        self._agents.append(agent)
        return agent.aid

    def step(self) -> list[tuple[int, float, float, float, float]]:
        """Advance one frame. Returns list of (id, x, y, ori_x, ori_y)."""
        out: list[tuple[int, float, float, float, float]] = []
        survivors: list[_Agent] = []
        for agent in self._agents:
            self._advance_agent(agent)
            agent.lifetime -= 1
            if agent.lifetime > 0:
                survivors.append(agent)
                out.append((
                    agent.aid,
                    agent.x,
                    agent.y,
                    math.cos(agent.heading),
                    math.sin(agent.heading),
                ))
        self._agents = survivors
        return out

    def _advance_agent(self, agent: _Agent) -> None:
        """One frame of the T&P Figure 8 decision loop.

        Order (continuous-space port of Figure 8):
          1. If n steps have been taken → reselect destination from FOV cone.
          2. Try a step toward the destination (along heading).
          3. If blocked → try a side step (±90° from heading).
          4. If both blocked → reselect destination omni-directionally
             (no FOV cone, per the "back to top" arrow in Figure 8) and
             try once more. If still blocked, the agent stays this frame.
        """
        if agent.steps_until_decision <= 0:
            self._redecide(agent, omni=False)

        if not self._try_step_then_side(agent):
            # Stuck: per T&P, reselect from the top (omni) and try again.
            self._redecide(agent, omni=True)
            self._try_step_then_side(agent)

        # Arrival shortcut: forces a re-decide on the next frame so the
        # agent doesn't walk past its target. T&P's grid loop reaches the
        # same outcome via step-not-possible → side-step → reselect; this
        # is the continuous-space equivalent.
        target_xy = self._grid.cells[agent.target_idx]
        if math.hypot(agent.x - target_xy[0], agent.y - target_xy[1]) < self._step_length:
            agent.steps_until_decision = 0

    def _try_step_then_side(self, agent: _Agent) -> bool:
        """T&P Figure 8: try step toward destination, else side step."""
        if self._take_step(agent, agent.heading):
            return True
        # Side step: ±90° from heading, random sign chosen first.
        sign = 1 if self._rng.random() < 0.5 else -1
        for s in (sign, -sign):
            if self._take_step(agent, agent.heading + s * math.pi / 2.0):
                return True
        return False

    def _take_step(self, agent: _Agent, direction: float) -> bool:
        """Move `step_length` in `direction` if the segment is wall-free."""
        nx = agent.x + self._step_length * math.cos(direction)
        ny = agent.y + self._step_length * math.sin(direction)
        if not self._segment_clear(agent.x, agent.y, nx, ny):
            return False
        agent.x = nx
        agent.y = ny
        agent.steps_until_decision -= 1
        return True

    def _segment_clear(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """True iff the line segment doesn't cross any wall."""
        if self._wall_tree is None:
            return True
        seg = LineString([(x1, y1), (x2, y2)])
        for w_idx in self._wall_tree.query(seg):
            if seg.crosses(self._walls[int(w_idx)]):
                return False
        return True

    def _redecide(self, agent: _Agent, *, omni: bool) -> None:
        """Pick a new target. `omni=True` skips the FOV cone (T&P top node)."""
        cell_idx = int(self._kdtree.query([agent.x, agent.y])[1])
        visible = self._grid.visible[cell_idx]
        if visible.size == 0:
            # Boxed in — no visible neighbours. Keep heading, defer decision.
            agent.steps_until_decision = self._params.steps_before_turn
            return

        if omni:
            candidates = visible
        else:
            cone = self._fov_filter(agent, visible)
            candidates = cone if cone.size > 0 else visible  # relax if empty
        # Uniform sampling per Turner & Penn (2002). Do NOT distance-weight:
        # the through-vision (sum of sight lines through a cell) is an
        # *analytical* statistic equivalent to the agent steady state, not
        # a sampling rule. The published EVA agent samples uniformly within
        # the FOV cone; biasing toward far targets would change which
        # cells the heatmap concentrates on.
        choice = int(self._rng.choice(candidates))
        agent.target_idx = choice
        target_xy = self._grid.cells[choice]
        agent.heading = math.atan2(target_xy[1] - agent.y, target_xy[0] - agent.x)
        agent.steps_until_decision = self._params.steps_before_turn

    def _fov_filter(self, agent: _Agent, visible: np.ndarray) -> np.ndarray:
        """Keep cells whose direction is within ±half_fov of agent's heading."""
        targets = self._grid.cells[visible]
        dx = targets[:, 0] - agent.x
        dy = targets[:, 1] - agent.y
        angles = np.arctan2(dy, dx)
        # Wrap delta into [-pi, pi] so |delta| <= half_fov is the cone test.
        delta = np.mod(angles - agent.heading + math.pi, 2.0 * math.pi) - math.pi
        return visible[np.abs(delta) <= self._half_fov]


def sample_uniform_in_grid(
    grid: VisibilityGrid, rng: np.random.Generator
) -> tuple[float, float]:
    """Sample one (x, y) by picking a random grid cell uniformly."""
    idx = int(rng.integers(0, len(grid.cells)))
    x, y = grid.cells[idx]
    return float(x), float(y)


def sample_uniform_in_polygons(
    polygons: list[Polygon], rng: np.random.Generator
) -> tuple[float, float]:
    """Rejection-sample one (x, y) uniformly across the union of polygons.

    Picks a polygon weighted by area, then rejection-samples inside its
    bounding box. Caps retries — falls back to the polygon's centroid if
    rejection fails repeatedly (shouldn't happen for non-degenerate inputs).
    """
    if not polygons:
        raise ValueError("no release polygons")
    areas = np.array([p.area for p in polygons], dtype=float)
    pick = int(rng.choice(len(polygons), p=areas / areas.sum()))
    poly = polygons[pick]
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(100):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        from shapely.geometry import Point as _Point
        if poly.contains(_Point(x, y)):
            return x, y
    c = poly.centroid
    return float(c.x), float(c.y)


def release_iter(
    rate_per_frame: float,
    rng: np.random.Generator,
) -> Iterator[int]:
    """Yield the number of agents to spawn each frame for a Poisson-ish rate.

    Uses the integer + fractional split (deterministic given rate): each frame
    spawns `floor(rate)` agents plus one extra with probability `rate - floor(rate)`.
    Cheap and matches depthmapX's effective behaviour for rate < 1.
    """
    base = math.floor(rate_per_frame)
    frac = rate_per_frame - base
    while True:
        extra = 1 if rng.random() < frac else 0
        yield base + extra
