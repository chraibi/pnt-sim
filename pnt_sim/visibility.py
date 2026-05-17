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
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

# Below this cell count, the multiprocessing overhead (spawn + per-worker
# pickling of cells/walls) costs more than the parallelism saves, so the
# serial path stays faster. Tuned empirically on a laptop-class machine.
_MP_THRESHOLD = 500


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
    workers: int | None = None,
) -> VisibilityGrid:
    """Build the visibility grid for a walkable polygon.

    `workers` controls the visibility precompute:
      - None (default): auto — use all CPUs for grids above the MP threshold,
        otherwise serial.
      - 1: force serial (useful for debugging or small grids).
      - N > 1: use N worker processes.
    """
    cells = _generate_grid_points(walkable, grid_spacing, min_wall_distance)
    walls = _wall_segments(walkable)
    visible = _compute_visibility(cells, walls, workers=workers)
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
    workers: int | None = None,
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

    vg = build_visibility(
        walkable,
        grid_spacing,
        min_wall_distance=min_wall_distance,
        workers=workers,
    )
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


# Worker-process state. multiprocessing initializers populate these once
# per worker so each task call doesn't re-pickle the cells/walls arrays.
_WORKER_STATE: dict[str, np.ndarray] = {}


def _init_worker(
    cells_arr: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    wdx: np.ndarray,
    wdy: np.ndarray,
) -> None:
    _WORKER_STATE["cells"] = cells_arr
    _WORKER_STATE["w1"] = w1
    _WORKER_STATE["w2"] = w2
    _WORKER_STATE["wdx"] = wdx
    _WORKER_STATE["wdy"] = wdy


def _row_visibility(
    i: int,
    cells_arr: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    wdx: np.ndarray,
    wdy: np.ndarray,
) -> np.ndarray:
    """Indices j > i visible from cell i. Pure-numpy CCW-orientation test."""
    p1 = cells_arr[i]
    p2s = cells_arr[i + 1 :]
    if p2s.size == 0:
        return np.empty(0, dtype=np.int32)

    # Segments cross iff each separates the other's endpoints. Use the raw
    # cross-product sign (d * d' < 0) rather than np.sign — same predicate
    # without the per-element sign call.
    dx = (p2s[:, 0] - p1[0])[:, None]
    dy = (p2s[:, 1] - p1[1])[:, None]
    d1 = dx * (w1[:, 1] - p1[1]) - dy * (w1[:, 0] - p1[0])
    d2 = dx * (w2[:, 1] - p1[1]) - dy * (w2[:, 0] - p1[0])
    d3 = wdx * (p1[1] - w1[:, 1]) - wdy * (p1[0] - w1[:, 0])
    d4 = wdx * (p2s[:, 1:2] - w1[:, 1]) - wdy * (p2s[:, 0:1] - w1[:, 0])

    # Proper-crossing test (matches shapely.crosses for 1D-on-1D):
    # strict sign disagreement on both segments — collinear / endpoint
    # touches don't count.
    blocked = ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)
    return np.flatnonzero(~blocked).astype(np.int32) + (i + 1)


def _worker_chunk(indices: list[int]) -> list[tuple[int, np.ndarray]]:
    cells_arr = _WORKER_STATE["cells"]
    w1 = _WORKER_STATE["w1"]
    w2 = _WORKER_STATE["w2"]
    wdx = _WORKER_STATE["wdx"]
    wdy = _WORKER_STATE["wdy"]
    return [
        (i, _row_visibility(i, cells_arr, w1, w2, wdx, wdy)) for i in indices
    ]


def _resolve_workers(n: int, workers: int | None) -> int:
    if workers is None:
        if n < _MP_THRESHOLD:
            return 1
        return max(1, os.cpu_count() or 1)
    return max(1, workers)


def _compute_visibility(
    cells: np.ndarray,
    walls: list[LineString],
    *,
    workers: int | None = None,
) -> list[np.ndarray]:
    """For each cell, return indices of other cells with unobstructed LOS."""
    n = len(cells)
    if not walls:
        # No walls — every cell can see every other cell. Pathological for
        # space syntax (uniform visitation), but mathematically defined.
        full = np.arange(n, dtype=np.int32)
        return [np.delete(full, i) for i in range(n)]

    # Wall endpoints. _wall_segments emits 2-point LineStrings, so each
    # row of (w1, w2) is one wall segment.
    w1 = np.asarray([w.coords[0] for w in walls], dtype=np.float64)
    w2 = np.asarray([w.coords[-1] for w in walls], dtype=np.float64)
    wdx = w2[:, 0] - w1[:, 0]
    wdy = w2[:, 1] - w1[:, 1]

    cells_arr = np.asarray(cells, dtype=np.float64)
    rows: list[list[int]] = [[] for _ in range(n)]
    total_pairs = n * (n - 1) // 2

    n_workers = _resolve_workers(n, workers)

    if n_workers == 1:
        with tqdm(
            total=total_pairs,
            desc=f"visibility ({n} cells)",
            unit="pair",
            unit_scale=True,
            mininterval=0.5,
        ) as bar:
            for i in range(n):
                j_arr = _row_visibility(i, cells_arr, w1, w2, wdx, wdy)
                for j in j_arr:
                    rows[i].append(int(j))
                    rows[int(j)].append(i)
                bar.update(n - 1 - i)
        return [np.asarray(r, dtype=np.int32) for r in rows]

    # Parallel path: chunk indices in a striped pattern so each chunk has
    # roughly the same total work (i=0 does N-1 candidates, i=N-1 does 0).
    # Striped distribution balances cost across workers without sorting.
    chunks_per_worker = 8
    total_chunks = max(1, n_workers * chunks_per_worker)
    chunks: list[list[int]] = [
        list(range(c, n, total_chunks)) for c in range(total_chunks)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(cells_arr, w1, w2, wdx, wdy),
    ) as pool, tqdm(
        total=total_pairs,
        desc=f"visibility ({n} cells, {n_workers} workers)",
        unit="pair",
        unit_scale=True,
        mininterval=0.5,
    ) as bar:
        for result in pool.imap_unordered(_worker_chunk, chunks):
            for i, j_arr in result:
                for j in j_arr:
                    rows[i].append(int(j))
                    rows[int(j)].append(i)
                bar.update(n - 1 - i)

    return [np.asarray(r, dtype=np.int32) for r in rows]
