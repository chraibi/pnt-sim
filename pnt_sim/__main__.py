"""CLI entry point for pnt-sim.

Wires scenario_io -> visibility (with cache) -> AgentSimulation -> TrajectoryWriter.

Usage:
    python -m pnt_sim --wkt geometry.wkt --out trails.sqlite \
        --grid 0.5 --fov-deg 170 --steps-before-turn 3 \
        --release-rate 0.1 --max-agents 50 --timesteps 5000 \
        --agent-lifetime 1000 --release-mode uniform --seed 42

    Pass --json scenario/config.json only when --release-mode distributions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .agents import (
    AgentParams,
    AgentSimulation,
    release_iter,
    sample_uniform_in_grid,
    sample_uniform_in_polygons,
)
from .scenario_io import load_scenario
from .visibility import load_or_build
from .writer import TrajectoryWriter


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pnt-sim",
        description="Penn & Turner space-syntax agent simulator.",
    )
    p.add_argument("--wkt", required=True, type=Path, help="scenario geometry.wkt")
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="scenario config.json (only needed for --release-mode distributions)",
    )
    p.add_argument("--out", required=True, type=Path, help="output sqlite path")
    p.add_argument("--grid", type=float, default=0.5, help="grid spacing in m")
    p.add_argument("--clearance", type=float, default=0.1, help="min wall distance for grid cells")
    p.add_argument("--fov-deg", type=float, default=170.0)
    p.add_argument("--steps-before-turn", type=int, default=3)
    p.add_argument("--step-length", type=float, default=None, help="default = grid spacing")
    p.add_argument("--release-rate", type=float, default=0.1, help="agents per frame")
    p.add_argument("--release-mode", choices=["uniform", "distributions"], default="uniform")
    p.add_argument("--max-agents", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=5000)
    p.add_argument("--agent-lifetime", type=int, default=1000)
    p.add_argument("--fps", type=float, default=10.0, help="frame rate stamped into metadata")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="visibility cache file (default: <out>.visibility.pkl)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="visibility precompute workers (default: auto — all CPUs for "
        "large grids, serial below ~500 cells). 1 forces serial.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    if args.release_mode == "distributions" and args.json is None:
        print(
            "error: --release-mode distributions requires --json (release zones "
            "are read from config.json)",
            file=sys.stderr,
        )
        return 2
    scenario = load_scenario(args.wkt, args.json)
    print(
        f"scenario loaded: walkable bounds={scenario.walkable.bounds}, "
        f"release_zones={len(scenario.release_zones)}",
        file=sys.stderr,
    )

    cache_path = args.cache or args.out.with_suffix(args.out.suffix + ".visibility.pkl")
    grid = load_or_build(
        scenario.walkable,
        scenario.walkable_wkt,
        args.grid,
        cache_path,
        min_wall_distance=args.clearance,
        workers=args.workers,
    )
    print(
        f"visibility grid: cells={len(grid.cells)}, "
        f"avg-visible={sum(v.size for v in grid.visible) / max(1, len(grid.visible)):.1f}, "
        f"cache={cache_path}",
        file=sys.stderr,
    )

    if args.release_mode == "distributions" and not scenario.release_zones:
        print(
            "release-mode=distributions but scenario has no distribution polygons; "
            "falling back to uniform release",
            file=sys.stderr,
        )
        release_mode = "uniform"
    else:
        release_mode = args.release_mode

    params = AgentParams(
        fov_deg=args.fov_deg,
        steps_before_turn=args.steps_before_turn,
        step_length=args.step_length,
        agent_lifetime=args.agent_lifetime,
    )
    sim = AgentSimulation(grid, params, rng)
    spawner = release_iter(args.release_rate, rng)
    zone_polys = [z.polygon for z in scenario.release_zones]

    t_compute_done = time.perf_counter()
    setup_s = t_compute_done - t0
    print(
        f"setup done in {setup_s:.1f}s "
        f"(visibility: {sum(v.size for v in grid.visible) // 2} pair-edges)",
        file=sys.stderr,
    )

    total_spawned = 0
    total_rows = 0
    with TrajectoryWriter(args.out, scenario.walkable_wkt, fps=args.fps) as writer:
        bar = tqdm(
            range(args.timesteps),
            desc="simulation",
            unit="frame",
            mininterval=0.5,
        )
        for frame in bar:
            # Spawn new agents up to the cap.
            to_spawn = next(spawner)
            for _ in range(to_spawn):
                if sim.alive_count >= args.max_agents:
                    break
                if release_mode == "distributions":
                    x, y = sample_uniform_in_polygons(zone_polys, rng)
                else:
                    x, y = sample_uniform_in_grid(grid, rng)
                sim.spawn(x, y)
                total_spawned += 1
            positions = sim.step()
            writer.write_frame(frame, positions)
            total_rows += len(positions)
            # Light postfix — only updated when tqdm decides to repaint
            # (mininterval=0.5s) so this isn't a per-frame cost.
            bar.set_postfix(alive=sim.alive_count, total=total_spawned, refresh=False)

    sim_s = time.perf_counter() - t_compute_done
    frames = args.timesteps
    print(
        f"\nperf summary:\n"
        f"  visibility precompute : {setup_s:8.2f} s "
        f"({len(grid.cells)} cells, {len(grid.cells)*(len(grid.cells)-1)//2:,} pairs, "
        f"{1e6 * setup_s / max(1, len(grid.cells)*(len(grid.cells)-1)//2):.2f} us/pair)\n"
        f"  simulation            : {sim_s:8.2f} s "
        f"({frames} frames, {frames/sim_s:.1f} frames/s, "
        f"{1e3 * sim_s / max(1, frames):.2f} ms/frame, "
        f"{total_rows:,} trajectory rows)\n"
        f"  total wall            : {setup_s + sim_s:8.2f} s\n"
        f"  output                : {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
