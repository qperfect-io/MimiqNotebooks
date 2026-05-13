# Copyright (c) 2026 QPerfect
# SPDX-License-Identifier: MIT
"""Strasbourg places — canonical data for the Quantum in Practice demo.

Single source of truth for the geography shared by the notebook, `tsp.py`,
and `vrp.py`. The 11 Christmas-market sites and 14 monuments mirror the
lists in Aymane's draft notebook (cell 8). Coordinates are WGS-84 decimal
degrees; distances use `geopy.distance.geodesic` (Karney's algorithm on
the WGS-84 ellipsoid).

Quick reference:

    >>> from strasbourg_markets_demo.strasbourg import STRASBOURG as s
    >>> s.find("kleber").name
    'Place Kléber'
    >>> s.distance_m("kleber", "broglie")          # accept keys
    422.5...
    >>> s.distance_m(s.find("kleber"), s.find("broglie"))   # or Place objects
    422.5...
    >>> [p.key for p in s.tsp_five()]
    ['kleber', 'broglie', 'cathedrale_market', 'chateau', 'marche_gayot']
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from geopy.distance import geodesic

PlaceKind = Literal["market", "monument", "transport"]
PlaceLike = "Place | str | tuple[float, float]"


@dataclass(frozen=True, slots=True)
class Place:
    """A geographic feature in central Strasbourg."""

    key: str
    name: str
    lat: float
    lon: float
    kind: PlaceKind
    emoji: str = ""
    color: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def coord(self) -> tuple[float, float]:
        return (self.lat, self.lon)

    def distance_m(self, other: "Place | str | tuple[float, float]") -> float:
        """Geodesic distance in metres to another place or `(lat, lon)` pair."""
        return STRASBOURG.distance_m(self, other)


# -- Christmas-market sites (Aymane notebook cell 8, XMAS_MARKETS) --------------

MARKETS: tuple[Place, ...] = (
    Place("kleber", "Place Kléber",
          48.58331, 7.74596, "market",
          emoji="🎄", color="#ff4757",
          description="Grand Sapin (30 m), ice rink, 130+ stands. "
                      "Christkindelsmärik historique.",
          aliases=("Kléber", "Place Kleber")),
    Place("broglie", "Place Broglie",
          48.58499, 7.75032, "market",
          emoji="⭐", color="#ffa502",
          description="Oldest market in France (since 1570). "
                      "Animated light projection on Hôtel de Ville.",
          aliases=("Broglie",)),
    Place("cathedrale_market", "Cathédrale Notre-Dame (marché)",
          48.58156, 7.75016, "market",
          emoji="🔔", color="#2ed573",
          description="Gothic cathedral backdrop. Pain d'épices, "
                      "local Alsatian wines.",
          aliases=("Cathédrale market", "Cathedrale market")),
    Place("chateau", "Place du Château",
          48.58134, 7.75139, "market",
          emoji="🏰", color="#1e90ff",
          description="Alsatian tradition market, next to Palais des Rohan.",
          aliases=("Château", "Place du Chateau", "Chateau")),
    Place("marche_gayot", "Place du Marché Gayot",
          48.58264, 7.75321, "market",
          emoji="🎨", color="#a29bfe",
          description="Intimate artisan market. Handmade crafts in a "
                      "side-street setting.",
          aliases=("Marché Gayot", "Marche Gayot", "Gayot")),
    Place("benjamin_zix", "Place Benjamin Zix",
          48.58132, 7.74241, "market",
          emoji="✨", color="#22d3ee",
          description="Petite France district. Meisenthal glass baubles, "
                      "half-timbered houses.",
          aliases=("Benjamin Zix",)),
    Place("temple_neuf", "Place du Temple Neuf",
          48.58316, 7.74786, "market",
          emoji="🎁", color="#f78fb3",
          description="Carré d'Or market. Artisan jewellery, "
                      "goldsmith district.",
          aliases=("Temple Neuf",)),
    Place("quai_delices", "Quai des Délices",
          48.5812, 7.7534, "market",
          emoji="🍷", color="#00b894",
          description="Gourmet market alongside the Ill. Regional products, "
                      "foie gras, local charcuterie.",
          aliases=("Quai des Delices",)),
    Place("vieux_marche_poissons", "Place du Vieux Marché aux Poissons",
          48.58046, 7.74959, "market",
          emoji="🐟", color="#fd9644",
          description="Historic fish-market square. Local bredele and "
                      "foie gras producers.",
          aliases=("Vieux Marché aux Poissons", "Marché aux Poissons")),
    Place("austerlitz", "Place d'Austerlitz",
          48.57796, 7.75336, "market",
          emoji="🕯", color="#e84393",
          description="Neighbourhood market. Candlemakers and potters. "
                      "Quiet alternative to central crowds.",
          aliases=("Austerlitz",)),
    Place("parvis_cathedrale", "Place de la Cathédrale (parvis)",
          48.5820, 7.7502, "market",
          emoji="🌟", color="#f9ca24",
          description="Stalls directly facing the cathedral façade. "
                      "Santons, crèches, Alsatian ornaments.",
          aliases=("Parvis", "Parvis de la Cathédrale")),
)


# -- Monuments and must-see places (Aymane notebook cell 8, MONUMENTS) ----------

MONUMENTS: tuple[Place, ...] = (
    Place("cathedrale_notre_dame", "Cathédrale Notre-Dame de Strasbourg",
          48.58189, 7.75105, "monument",
          emoji="⛪", color="#c8a951",
          description="UNESCO World Heritage. Pink sandstone Gothic. "
                      "Astronomical clock at 12:30. Tallest in Christendom "
                      "until 1874.",
          aliases=("Cathédrale", "Cathedrale Notre-Dame", "Cathedral")),
    Place("palais_rohan", "Palais des Rohan",
          48.58097, 7.75228, "monument",
          emoji="🏛", color="#c8a951",
          description="18th-c. cardinal's palace. Three museums: Beaux-Arts, "
                      "Décoratifs, Archéologique.",
          aliases=("Palais Rohan", "Rohan")),
    Place("maison_kammerzell", "Maison Kammerzell",
          48.58197, 7.74975, "monument",
          emoji="🏚", color="#c8a951",
          description="Finest late-Gothic timber house in Alsace (1467). "
                      "Ornately carved façade.",
          aliases=("Kammerzell",)),
    Place("ponts_couverts", "Ponts Couverts & Tours médiévales",
          48.58005, 7.73937, "monument",
          emoji="🌉", color="#c8a951",
          description="Three 14th-c. bridges, four medieval guard towers. "
                      "Best panoramic viewpoint in Strasbourg.",
          aliases=("Ponts Couverts", "Covered Bridges")),
    Place("barrage_vauban", "Barrage Vauban",
          48.57960, 7.73800, "monument",
          emoji="🏗", color="#c8a951",
          description="17th-c. military dam by Vauban. Free rooftop terrace "
                      "with 360° view over Petite France.",
          aliases=("Vauban",)),
    Place("petite_france", "La Petite France",
          48.58136, 7.74218, "monument",
          emoji="🏘", color="#c8a951",
          description="UNESCO-listed medieval tanners' quarter. "
                      "Half-timbered houses, canals.",
          aliases=("Petite France",)),
    Place("saint_thomas", "Église Saint-Thomas",
          48.57982, 7.74548, "monument",
          emoji="🎵", color="#c8a951",
          description="Collegiate church with famous Silbermann organ (1741). "
                      "Mausoleum of Marshal de Saxe.",
          aliases=("Saint-Thomas", "Saint Thomas")),
    Place("place_gutenberg", "Place Gutenberg",
          48.58119, 7.74854, "monument",
          emoji="📖", color="#c8a951",
          description="Central square with statue of Johannes Gutenberg. "
                      "Hub between Kléber and the cathedral.",
          aliases=("Gutenberg",)),
    Place("musee_alsacien", "Musée Alsacien",
          48.57913, 7.75060, "monument",
          emoji="🏺", color="#c8a951",
          description="Folk museum in three Renaissance townhouses. "
                      "Traditional Alsatian interiors.",
          aliases=("Musee Alsacien",)),
    Place("place_republique", "Place de la République",
          48.58650, 7.75493, "monument",
          emoji="🌳", color="#c8a951",
          description="Grand Wilhelminian square (German imperial era). "
                      "National Theatre, Préfecture.",
          aliases=("République", "Place de la Republique")),
    Place("saint_pierre_jeune", "Église Saint-Pierre-le-Jeune Protestant",
          48.5841, 7.7466, "monument",
          emoji="🕍", color="#c8a951",
          description="12th–14th c. Gothic church with medieval cloister.",
          aliases=("Saint-Pierre-le-Jeune", "Saint Pierre le Jeune")),
    Place("mamcs", "Musée d'Art Moderne et Contemporain (MAMCS)",
          48.57948, 7.73610, "monument",
          emoji="🎭", color="#c8a951",
          description="Riverfront modern-art museum. Works from 1870 to "
                      "present.",
          aliases=("MAMCS", "Musée d'Art Moderne", "Musee Art Moderne")),
    Place("grande_ile", "Grande Île — UNESCO Zone",
          48.58296, 7.74783, "monument",
          emoji="🗺", color="#c8a951",
          description="First city centre to be UNESCO-listed (1988). "
                      "All markets within it.",
          aliases=("Grande Ile", "UNESCO")),
    Place("orfevres_merciere", "Rue des Orfèvres & Rue Mercière",
          48.58256, 7.74896, "monument",
          emoji="🛍", color="#c8a951",
          description="Strasbourg's main luxury shopping streets. Jewellers, "
                      "chocolatiers, festive window displays.",
          aliases=("Rue des Orfèvres", "Rue Mercière")),
)


# -- Transport hubs ------------------------------------------------------------

TRANSPORT: tuple[Place, ...] = (
    Place("gare_centrale", "Gare de Strasbourg",
          48.58431, 7.73472, "transport",
          emoji="🚆", color="#5a6268",
          description="Strasbourg main railway station. Default depot for "
                      "the Vehicle Routing Problem.",
          aliases=("Gare Centrale", "Strasbourg Gare", "Main Station")),
)


ALL_PLACES: tuple[Place, ...] = MARKETS + MONUMENTS + TRANSPORT


def _normalize(s: str) -> str:
    """Lowercase, strip diacritics and non-alphanumerics — for fuzzy lookup."""
    decomposed = unicodedata.normalize("NFKD", s)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in ascii_only.lower() if c.isalnum())


_BY_KEY: dict[str, Place] = {p.key: p for p in ALL_PLACES}
_BY_NORMAL: dict[str, Place] = {}
for _p in ALL_PLACES:
    for _label in (_p.key, _p.name, *_p.aliases):
        _BY_NORMAL[_normalize(_label)] = _p


# -- Legend helper -------------------------------------------------------------

def place_legend_outside(ax, *, side: str = "right",
                         title: str | None = None,
                         fontsize: int = 9):
    """Place `ax`'s legend outside the axes box.

    `side` is one of `"right"` (default), `"left"`, or `"bottom"`.
    Returns the `Legend` artist, or `None` if there's nothing to label.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if side == "right":
        loc, anchor, ncol = "center left", (1.02, 0.5), 1
    elif side == "left":
        loc, anchor, ncol = "center right", (-0.02, 0.5), 1
    elif side == "bottom":
        loc, anchor = "upper center", (0.5, -0.12)
        ncol = min(len(handles), 4)
    else:
        raise ValueError(f"unknown side {side!r}; use right|left|bottom")
    return ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor,
                     borderaxespad=0.0, fontsize=fontsize, ncol=ncol,
                     title=title, frameon=True)


# -- The Strasbourg façade -----------------------------------------------------

# Demo subset for Aymane's notebook (cell 12). Order is fixed: index 0 is
# Place Kléber, and downstream code may rely on that.
_TSP_FIVE_KEYS: tuple[str, ...] = (
    "kleber", "broglie", "cathedrale_market", "chateau", "marche_gayot",
)


class Strasbourg:
    """Christmas-market geography of Strasbourg with WGS-84 geodesic distances.

    Stateless; all data lives in module-level constants (`MARKETS`,
    `MONUMENTS`, `TRANSPORT`). Use the default singleton `STRASBOURG` or
    instantiate your own:

        from strasbourg_markets_demo.strasbourg import STRASBOURG as s
        s.find("kleber")
        s.distance_m("kleber", "broglie")
    """

    markets: tuple[Place, ...] = MARKETS
    monuments: tuple[Place, ...] = MONUMENTS
    transport: tuple[Place, ...] = TRANSPORT
    all_places: tuple[Place, ...] = ALL_PLACES

    # --- lookup ---------------------------------------------------------------

    def get(self, key: str) -> Place:
        """Return the place with this exact `key` slug. KeyError if unknown."""
        return _BY_KEY[key]

    def find(self, query: str) -> Place:
        """Resolve a place by key, name, or alias (case- and accent-insensitive)."""
        norm = _normalize(query)
        if norm in _BY_NORMAL:
            return _BY_NORMAL[norm]
        raise KeyError(f"no Strasbourg place matches {query!r}; "
                       f"known keys: {sorted(_BY_KEY)}")

    def by_kind(self, kind: PlaceKind) -> tuple[Place, ...]:
        """Every place with the given `kind`."""
        return tuple(p for p in ALL_PLACES if p.kind == kind)

    # --- distances (geopy / WGS-84 geodesic) ---------------------------------

    def _coord(self, p: "Place | str | tuple[float, float]") -> tuple[float, float]:
        if isinstance(p, Place):
            return p.coord
        if isinstance(p, str):
            return self.find(p).coord
        return (float(p[0]), float(p[1]))

    def distance_m(self,
                   a: "Place | str | tuple[float, float]",
                   b: "Place | str | tuple[float, float]") -> float:
        """Geodesic distance in metres between two places (or (lat, lon) pairs)."""
        return geodesic(self._coord(a), self._coord(b)).meters

    def distance_matrix_m(
        self,
        places: Sequence["Place | str | tuple[float, float]"],
    ) -> np.ndarray:
        """Pairwise geodesic distance matrix in metres, shape `(n, n)`."""
        coords = [self._coord(p) for p in places]
        n = len(coords)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = geodesic(coords[i], coords[j]).meters
                D[i, j] = d
                D[j, i] = d
        return D

    def coords_array(
        self,
        places: Sequence["Place | str"],
    ) -> np.ndarray:
        """`(n, 2)` ndarray of `(lat, lon)` pairs."""
        return np.array([self._coord(p) for p in places], dtype=float)

    # --- plotting -------------------------------------------------------------

    _MARKER_BY_KIND: dict[str, str] = {
        "market": "o",
        "monument": "D",
        "transport": "s",
    }

    def plot(
        self,
        places: Sequence["Place | str"] | None = None,
        edges: Sequence[tuple] | None = None,
        *,
        ax=None,
        figsize: tuple[float, float] = (10, 8),
        show_labels: bool = True,
        label: str = "name",          # "name" | "key" | "emoji" | "none"
        marker_size: float = 220.0,
        edge_color: str = "#ffa502",
        edge_width: float = 2.0,
        edge_alpha: float = 0.85,
        edge_style: str = "-",
        basemap: bool = False,
        basemap_provider: str | None = None,
        title: str | None = None,
        margin: float = 0.10,
        legend: bool | str = False,
    ):
        """Plot a set of places on a matplotlib axes, with optional edges.

        `places` defaults to the markets. `edges` is an iterable of `(a, b)`
        pairs — each `a`/`b` may be a `Place`, a key string, or a name. Edges
        are drawn as straight lines (great-circle is a no-op at this scale).

        With `basemap=True`, fetches an OSM raster basemap via the bundled
        `basemap` module. Provider names live in `basemap.PROVIDERS`; the
        default is `"carto_darkmatter"`. Tiles are cached in
        `~/.cache/strasbourg_markets_demo/tiles/`.

        If `legend` is True (or a string, used as title), the kind legend is
        placed outside the axes on the right.
        """
        import matplotlib.pyplot as plt

        place_list: list[Place] = [
            p if isinstance(p, Place) else self.find(p)
            for p in (places if places is not None else MARKETS)
        ]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Draw edges first so markers paint on top.
        if edges:
            for a, b in edges:
                pa = a if isinstance(a, Place) else self.find(a)
                pb = b if isinstance(b, Place) else self.find(b)
                ax.plot([pa.lon, pb.lon], [pa.lat, pb.lat],
                        color=edge_color, linewidth=edge_width,
                        alpha=edge_alpha, linestyle=edge_style, zorder=2)

        # Group markers by kind so the legend gets one entry per kind.
        for kind in ("market", "monument", "transport"):
            sub = [p for p in place_list if p.kind == kind]
            if not sub:
                continue
            ax.scatter(
                [p.lon for p in sub], [p.lat for p in sub],
                s=marker_size,
                c=[p.color or "#ffa502" for p in sub],
                marker=self._MARKER_BY_KIND[kind],
                edgecolors="white", linewidths=1.5,
                label=kind, zorder=4,
            )

        if show_labels and label != "none":
            for p in place_list:
                text = (
                    p.name if label == "name"
                    else p.key if label == "key"
                    else p.emoji if label == "emoji"
                    else p.name
                )
                ax.annotate(
                    text, (p.lon, p.lat),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=9, zorder=5,
                )

        # Auto-zoom to the place set with `margin` padding (fraction of span).
        lons = [p.lon for p in place_list]
        lats = [p.lat for p in place_list]
        if lons and lats:
            dlon = max(max(lons) - min(lons), 1e-3)
            dlat = max(max(lats) - min(lats), 1e-3)
            ax.set_xlim(min(lons) - margin * dlon, max(lons) + margin * dlon)
            ax.set_ylim(min(lats) - margin * dlat, max(lats) + margin * dlat)

        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lats) if lats else 48.58)))
        if title:
            ax.set_title(title)

        if basemap:
            try:
                from . import basemap as _bm
                _bm.add_basemap(ax, provider=(basemap_provider
                                              or "carto_darkmatter"))
            except Exception as exc:
                import warnings
                warnings.warn(f"basemap=True but tile fetch failed ({exc}); "
                              "rendering without it.", stacklevel=2)

        if legend:
            place_legend_outside(ax,
                                 title=legend if isinstance(legend, str) else None)

        return ax

    # --- demo subsets ---------------------------------------------------------

    def tsp_five(self) -> tuple[Place, ...]:
        """The five Christmas markets used in the TSP demo."""
        return tuple(_BY_KEY[k] for k in _TSP_FIVE_KEYS)

    def vrp_default(self) -> tuple[Place, tuple[Place, ...]]:
        """Return `(depot, customers)` for the VRP demo.

        Depot is the Gare Centrale; customers are the five TSP markets plus
        Petite France — a geographically separated sixth stop that makes the
        optimal vehicle partition non-trivial.
        """
        depot = _BY_KEY["gare_centrale"]
        customers = (
            _BY_KEY["broglie"],
            _BY_KEY["kleber"],
            _BY_KEY["cathedrale_market"],
            _BY_KEY["chateau"],
            _BY_KEY["petite_france"],
        )
        return depot, customers


# Default singleton.
STRASBOURG = Strasbourg()


# Module-level function aliases.
get = STRASBOURG.get
find = STRASBOURG.find
by_kind = STRASBOURG.by_kind
distance_m = STRASBOURG.distance_m
distance_matrix_m = STRASBOURG.distance_matrix_m
coords_array = STRASBOURG.coords_array
tsp_five = STRASBOURG.tsp_five
vrp_default = STRASBOURG.vrp_default
plot = STRASBOURG.plot


__all__ = [
    "Place",
    "PlaceKind",
    "Strasbourg",
    "STRASBOURG",
    "MARKETS",
    "MONUMENTS",
    "TRANSPORT",
    "ALL_PLACES",
    "get",
    "find",
    "by_kind",
    "distance_m",
    "distance_matrix_m",
    "coords_array",
    "tsp_five",
    "vrp_default",
    "plot",
    "place_legend_outside",
]
