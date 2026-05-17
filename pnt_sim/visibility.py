"""Grid + per-cell visibility for Penn & Turner agents.

Discretizes the walkable area into a Cartesian grid (one cell per grid
intersection, like depthmapX's VGA grid), then computes for each cell the
set of other cells it can see (line of sight not blocked by walls or
obstacles). Cached to disk keyed on geometry-hash + grid spacing.

The agent loop reads `cells` (Nx2 numpy array) for positions and
`visible[i]` for each cell's visible-set indices.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm


@dataclass
class VisibilityGrid:
    cells: np.ndarray  # (N, 2) cell center coordinates
    visible: list[np.ndarray]  # visible[i] = indices visible from cell i
    grid_spacing: float
    bounds: tuple[float, float, float, float]
    walls: list[LineString]  # exterior + hole segments; agents use these
                             # to check whether their next step is blocked.


def build_visibility(
    walkable: Polygon,
    grid_spacing: float,
    *,
    min_wall_distance: float = 0.1,
) -> VisibilityGrid:
    cells = _generate_grid_points(walkable, grid_spacing, min_wall_distance)
    walls = _wall_segments(walkable)
    visible = _compute_visibility(cells, walls)
    return VisibilityGrid(
        cells=cells,
        visible=visible,
        grid_spacing=grid_spacing,
        bounds=walkable.bounds,
        walls=walls,
    )


def load_or_build(
    walkable: Polygon,
    walkable_wkt: str,
    grid_spacing: float,
    cache_path: Path | None,
    *,
    min_wall_distance: float = 0.1,
) -> VisibilityGrid:
    """Build or load a cached visibility grid.

    Cache key is geometry-WKT-sha + grid spacing + min-wall-distance. If
    `cache_path` is None or missing or stale, rebuild and write.
    """
    key = _cache_key(walkable_wkt, grid_spacing, min_wall_distance)
    if cache_path and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("key") == key:
                vg = payload["grid"]
                if isinstance(vg, VisibilityGrid):
                    return vg
        except Exception:
            # Stale or unreadable cache — silently rebuild rather than crash.
            pass

    vg = build_visibility(walkable, grid_spacing, min_wall_distance=min_wall_distance)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"key": key, "grid": vg}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return vg


def _cache_key(wkt: str, grid_spacing: float, min_wall_distance: float) -> str:
    h = hashlib.sha256()
    h.update(wkt.encode("utf-8"))
    h.update(
        f"|grid={grid_spacing}|clearance={min_wall_distance}|v=2".encode("utf-8")
    )
    return h.hexdigest()


def _generate_grid_points(
    walkable: Polygon,
    grid_spacing: float,
    min_wall_distance: float,
) -> np.ndarray:
    minx, miny, maxx, maxy = walkable.bounds
    xs = np.arange(minx, maxx + grid_spacing, grid_spacing)
    ys = np.arange(miny, maxy + grid_spacing, grid_spacing)

    pts: list[tuple[float, float]] = []
    interiors = [Polygon(h) for h in walkable.interiors]
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if not walkable.contains(pt):
                continue
            if walkable.exterior.distance(pt) < min_wall_distance:
                continue
            if any(h.distance(pt) < min_wall_distance for h in interiors):
                continue
            pts.append((float(x), float(y)))
    if not pts:
        raise ValueError("no grid cells fit in the walkable area")
    return np.asarray(pts, dtype=float)


def _wall_segments(walkable: Polygon) -> list[LineString]:
    rings: list[Iterable] = [walkable.exterior.coords]
    rings.extend(h.coords for h in walkable.interiors)
    segments: list[LineString] = []
    for coords in rings:
        pts = list(coords)
        for a, b in zip(pts[:-1], pts[1:]):
            segments.append(LineString([a, b]))
    return segments


def _compute_visibility(
    cells: np.ndarray,
    walls: list[LineString],
) -> list[np.ndarray]:
    """For each cell, return indices of other cells with unobstructed LOS.

    Vectorized segment-segment intersection: for each cell `i`, test all
    candidate target cells `j > i` against every wall in one numpy call
    using the standard CCW-orientation test. ~50× faster than the
    per-pair shapely loop on geometries with thousands of cells.
    """
    n = len(cells)
    if not walls:
        # No walls — every cell can see every other cell. Pathological for
        # space syntax (uniform visitation), but mathematically defined.
        full = np.arange(n, dtype=np.int32)
        return [np.delete(full, i) for i in range(n)]

    # Wall endpoints. _wall_segments emits 2-point LineStrings, so each
    # row of (w1, w2) is one wall segment.
    w1 = np.asarray([w.coords[0] for w in walls], dtype=np.float64)  # (K, 2)
    w2 = np.asarray([w.coords[-1] for w in walls], dtype=np.float64)  # (K, 2)
    wdx = w2[:, 0] - w1[:, 0]  # (K,)
    wdy = w2[:, 1] - w1[:, 1]

    cells_arr = np.asarray(cells, dtype=np.float64)
    rows: list[list[int]] = [[] for _ in range(n)]

    total_pairs = n * (n - 1) // 2
    with tqdm(
        total=total_pairs,
        desc=f"visibility ({n} cells)",
        unit="pair",
        unit_scale=True,
        mininterval=0.5,
    ) as bar:
        for i in range(n):
            p1 = cells_arr[i]
            p2s = cells_arr[i + 1 :]  # (M, 2)
            m = len(p2s)
            if m == 0:
                continue

            # Segments cross iff each separates the other's endpoints.
            # orient((a,b), c) = sign((b-a) x (c-a)). Use the raw product
            # rather than np.sign — `d1*d2 < 0` is the same predicate and
            # avoids the per-element sign call.
            dx = (p2s[:, 0] - p1[0])[:, None]  # (M, 1)
            dy = (p2s[:, 1] - p1[1])[:, None]
            d1 = dx * (w1[:, 1] - p1[1]) - dy * (w1[:, 0] - p1[0])  # (M, K)
            d2 = dx * (w2[:, 1] - p1[1]) - dy * (w2[:, 0] - p1[0])  # (M, K)
            d3 = wdx * (p1[1] - w1[:, 1]) - wdy * (p1[0] - w1[:, 0])  # (K,)
            d4 = (
                wdx * (p2s[:, 1:2] - w1[:, 1])
                - wdy * (p2s[:, 0:1] - w1[:, 0])
            )  # (M, K)

            # Proper-crossing test (matches shapely.crosses for 1D-on-1D):
            # strict sign disagreement on *both* segments — collinear /
            # endpoint touches don't count, exactly as in the old code.
            blocked = ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)  # (M,)

            visible_local = np.flatnonzero(~blocked)
            for j_local in visible_local:
                j = int(j_local) + i + 1
                rows[i].append(j)
                rows[j].append(i)

            bar.update(m)

    return [np.asarray(r, dtype=np.int32) for r in rows]
