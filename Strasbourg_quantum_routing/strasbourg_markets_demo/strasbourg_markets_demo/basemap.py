# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""Pure-Python OSM raster-tile basemap fetcher.

Stdlib `urllib` for HTTP, Pillow for stitching. Tiles cached in
`~/.cache/strasbourg_markets_demo/tiles/`.

Public API:
    fetch_basemap(bbox, provider, zoom, cache_dir) -> (img, extent, attribution)
    add_basemap(ax, provider, ...)                 -> ax  (uses ax xlim/ylim)
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

PROVIDERS: dict[str, tuple[str, str]] = {
    "carto_darkmatter": (
        "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors  © CARTO",
    ),
    "carto_positron": (
        "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors  © CARTO",
    ),
    "osm": (
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors",
    ),
}

USER_AGENT = "strasbourg_markets_demo/0.1 (+https://qperfect.io)"
DEFAULT_CACHE = Path.home() / ".cache" / "strasbourg_markets_demo" / "tiles"


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _pick_zoom(lat_min: float, lat_max: float,
               lon_min: float, lon_max: float,
               max_tiles_wide: int = 4) -> int:
    """Highest zoom where the bbox spans <= `max_tiles_wide` tiles."""
    for z in range(18, 0, -1):
        x0, _ = _lonlat_to_tile(lon_min, lat_max, z)
        x1, _ = _lonlat_to_tile(lon_max, lat_min, z)
        if x1 - x0 <= max_tiles_wide:
            return z
    return 1


def _download(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def fetch_basemap(
    bbox: tuple[float, float, float, float],
    *,
    provider: str = "carto_darkmatter",
    zoom: int | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    """Fetch and stitch tiles covering `bbox = (lat_min, lat_max, lon_min, lon_max)`.

    Returns `(image_array, extent, attribution)` where `extent` is
    `(lon_min, lon_max, lat_min, lat_max)` of the stitched image (slightly
    wider than the requested bbox due to whole-tile rounding).
    """
    from PIL import Image

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"choose from {list(PROVIDERS)}")
    url_tmpl, attribution = PROVIDERS[provider]
    lat_min, lat_max, lon_min, lon_max = bbox

    if zoom is None:
        zoom = _pick_zoom(lat_min, lat_max, lon_min, lon_max)

    x0_f, y1_f = _lonlat_to_tile(lon_min, lat_min, zoom)
    x1_f, y0_f = _lonlat_to_tile(lon_max, lat_max, zoom)
    x0, x1 = int(math.floor(x0_f)), int(math.floor(x1_f))
    y0, y1 = int(math.floor(y0_f)), int(math.floor(y1_f))

    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    canvas = Image.new("RGB", (nx * 256, ny * 256))
    cache_dir = Path(cache_dir)

    for ix in range(nx):
        for iy in range(ny):
            x, y = x0 + ix, y0 + iy
            url = url_tmpl.format(z=zoom, x=x, y=y)
            cache_path = cache_dir / provider / f"{zoom}_{x}_{y}.png"
            data = _download(url, cache_path)
            tile = Image.open(io.BytesIO(data)).convert("RGB")
            canvas.paste(tile, (ix * 256, iy * 256))

    lon_w, lat_n = _tile_to_lonlat(x0, y0, zoom)
    lon_e, lat_s = _tile_to_lonlat(x1 + 1, y1 + 1, zoom)
    return np.asarray(canvas), (lon_w, lon_e, lat_s, lat_n), attribution


def add_basemap(ax, *,
                provider: str = "carto_darkmatter",
                zoom: int | None = None,
                cache_dir: Path = DEFAULT_CACHE,
                attribution: bool = True):
    """Add a raster basemap to `ax`, using its current xlim/ylim as the bbox."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = (ylim[0], ylim[1], xlim[0], xlim[1])
    img, extent, attrib = fetch_basemap(
        bbox, provider=provider, zoom=zoom, cache_dir=cache_dir,
    )
    ax.imshow(img, extent=extent, origin="upper",
              interpolation="bilinear", zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if attribution:
        ax.text(0.99, 0.01, attrib, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=2))
    return ax


__all__ = ["PROVIDERS", "DEFAULT_CACHE",
           "fetch_basemap", "add_basemap"]
