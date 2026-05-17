"""Parse a Web-Based JuPedSim scenario into pnt-sim's input objects.

The web app saves two files when you "Save Scenario":
  - config.json   — exits / distributions / checkpoints / zones / journeys
  - geometry.wkt  — the walkable area as POLYGON((outer), (hole), ...) where
                    holes are obstacles.

For Penn & Turner we only need:
  - the walkable polygon (with holes), and
  - optionally, the distribution polygons as named release zones.

Everything else in config.json (transitions, journeys, exit parameters) is
ignored — Turner agents are target-less.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shapely import from_wkt
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


@dataclass
class ReleaseZone:
    """A named polygon agents can be spawned inside."""

    name: str
    polygon: Polygon


@dataclass
class Scenario:
    walkable: Polygon
    walkable_wkt: str
    release_zones: list[ReleaseZone] = field(default_factory=list)


def load_scenario(json_path: Path, wkt_path: Path) -> Scenario:
    """Load and validate a scenario from app-format files."""
    config = json.loads(Path(json_path).read_text(encoding="utf-8"))
    wkt_text = Path(wkt_path).read_text(encoding="utf-8").strip()

    walkable = _load_walkable(wkt_text)
    zones = _load_release_zones(config)
    return Scenario(walkable=walkable, walkable_wkt=wkt_text, release_zones=zones)


def _load_walkable(wkt_text: str) -> Polygon:
    if not wkt_text:
        raise ValueError("geometry WKT is empty")
    geom: BaseGeometry = from_wkt(wkt_text)
    if geom.is_empty:
        raise ValueError("geometry WKT is empty")
    if geom.geom_type == "Polygon":
        return geom  # type: ignore[return-value]
    if geom.geom_type == "MultiPolygon":
        # App always saves a single POLYGON; tolerate a MultiPolygon by
        # taking the largest piece so an out-of-tool-produced scenario
        # doesn't crash here.
        pieces = list(geom.geoms)  # type: ignore[attr-defined]
        return max(pieces, key=lambda p: p.area)
    raise ValueError(f"unsupported geometry type: {geom.geom_type}")


def _load_release_zones(config: dict) -> list[ReleaseZone]:
    zones: list[ReleaseZone] = []
    distributions = config.get("distributions") or {}
    if not isinstance(distributions, dict):
        return zones
    for name, dist in distributions.items():
        if not isinstance(dist, dict):
            continue
        coords = dist.get("coordinates")
        if not coords or len(coords) < 3:
            continue
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 0:
            continue
        zones.append(ReleaseZone(name=str(name), polygon=poly))
    return zones
