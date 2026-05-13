# Copyright (c) 2026 QPerfect
# SPDX-License-Identifier: MIT
"""QPerfect plot theme — a thin wrapper over matplotlib's style system.

The visual identity is split in two:

* `qperfect_dark.mplstyle` (sibling file): dark backgrounds, default
  qualitative color cycle, larger bold titles. Lives outside Python so it
  composes with any other matplotlib style sheet.
* `PALETTE` (this module): the named accent colors. Use these when you need
  to refer to a series by meaning ("the cyan one") rather than by cycle
  position. Same hex values that drive the Folium markers in
  `strasbourg.MARKETS[i].color`.

Quick reference:

    from strasbourg_markets_demo import theme
    theme.apply()                          # global, persists for the session

    with theme.context():                  # scoped — undoes itself on exit
        fig, ax = plt.subplots()
        ax.plot(x, y, color=theme.PALETTE["cyan"])

    theme.apply(["dark_background", theme.STYLE_FILE])    # compose styles
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt


_EMOJI_FONT_CANDIDATES: tuple[str, ...] = (
    # Listed in preference order; only installed names enter the fallback chain.
    "Noto Color Emoji",       # Linux
    "Apple Color Emoji",      # macOS
    "Segoe UI Emoji",         # Windows
    "Twemoji Mozilla",        # Firefox / some Linux packages
    "Symbola",                # monochrome catch-all
)


def _install_font_fallback() -> list[str]:
    """Set matplotlib's `font.family` to a chain that includes any installed
    emoji font. Returns the resolved chain.
    """
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    chain = ["DejaVu Sans"] + [name for name in _EMOJI_FONT_CANDIDATES
                               if name in installed]
    mpl.rcParams["font.family"] = chain
    return chain


def _silence_glyph_warnings() -> None:
    """Silence matplotlib's 'Glyph N missing from font' warnings and
    `findfont` log messages — safety net for hosts with no colour-emoji font.
    """
    import logging
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Glyph \d+.* missing from font.*",
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Path to the `.mplstyle` shipped in this package, resolved via
# importlib.resources (works for editable installs, wheels, and zip imports).
STYLE_FILE: Path = Path(str(files(__package__) / "qperfect_dark.mplstyle"))


# Named accent palette — matches Aymane's notebook THEME dict and the
# per-place `color` attributes in `strasbourg.MARKETS`.
#
#     ax.plot(x, y, color=PALETTE["cyan"])
PALETTE: dict[str, str] = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "fg":     "#e6edf3",
    "muted":  "#8b949e",
    "red":    "#ff4757",
    "green":  "#2ed573",
    "blue":   "#1e90ff",
    "yellow": "#ffa502",
    "purple": "#a29bfe",
    "cyan":   "#22d3ee",
    "pink":   "#f78fb3",
}


def apply(extra: str | list[str | Path] | None = None) -> None:
    """Apply the QPerfect dark style globally (persists until reset).

    Pass `extra` to compose with other matplotlib styles, e.g.
    `theme.apply("dark_background")`. The QPerfect style is applied last
    so it wins on conflicts. Also configures the emoji font fallback
    chain and silences residual 'Glyph missing' warnings.
    """
    styles: list[str | Path] = []
    if extra is not None:
        styles.extend([extra] if isinstance(extra, (str, Path)) else extra)
    styles.append(STYLE_FILE)
    plt.style.use(styles)
    _install_font_fallback()
    _silence_glyph_warnings()


@contextmanager
def context(extra: str | list[str | Path] | None = None) -> Iterator[None]:
    """Apply the QPerfect dark style for the duration of a `with` block."""
    styles: list[str | Path] = []
    if extra is not None:
        styles.extend([extra] if isinstance(extra, (str, Path)) else extra)
    styles.append(STYLE_FILE)
    with plt.style.context(styles), warnings.catch_warnings():
        _install_font_fallback()
        _silence_glyph_warnings()
        yield


__all__ = ["PALETTE", "STYLE_FILE", "apply", "context"]
