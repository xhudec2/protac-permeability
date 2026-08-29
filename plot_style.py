"""
Plot style — shared palette and matplotlib configuration.

Usage:
    from plot_style import *
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_RCPARAMS)

See the plot style guide for full documentation and examples.
From https://ailab.bio/
"""

import subprocess

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── Palette ─────────────────────────────────────────────
DEEP_TEAL = "#2D4F54"
ROYAL_PURPLE = "#7B539E"
SOFT_LAVENDER = "#B8A0D2"
FOREST_GREEN = "#5A9448"
MUTED_PLUM = "#9E4A78"
DARK_SLATE = "#3A3D4A"

PLOT_PALETTE = [
    DEEP_TEAL,
    ROYAL_PURPLE,
    SOFT_LAVENDER,
    FOREST_GREEN,
    MUTED_PLUM,
    DARK_SLATE,
]

# ── Matplotlib rcParams ─────────────────────────────────
PLOT_RCPARAMS = {
    # Typography — Latin Modern Roman, the default LaTeX body font used
    # in the plot paper template (plot.cls loads no font-substitution
    # package, so pdflatex renders Computer/Latin Modern).
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    # Axes — full bounding box (all four spines)
    "axes.facecolor": "white",
    "axes.edgecolor": DARK_SLATE,
    "axes.linewidth": 1.2,
    "axes.labelcolor": DARK_SLATE,
    "axes.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.prop_cycle": plt.cycler("color", PLOT_PALETTE),
    "axes.grid": False,
    # Ticks
    "xtick.color": DARK_SLATE,
    "ytick.color": DARK_SLATE,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.labelcolor": DARK_SLATE,
    # Figure & saving
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # Lines & markers
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
}


def _register_latin_modern():
    """Register the Latin Modern Roman TeX fonts with matplotlib, if a TeX install is found."""
    for name in [
        "lmroman10-regular.otf",
        "lmroman10-bold.otf",
        "lmroman10-italic.otf",
        "lmroman10-bolditalic.otf",
    ]:
        try:
            path = subprocess.check_output(["kpsewhich", name], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if path:
            fm.fontManager.addfont(path)


def apply_style():
    """Apply the PLOT rcParams to the current matplotlib session."""
    _register_latin_modern()
    plt.rcParams.update(PLOT_RCPARAMS)


def add_panel_label(ax, label, x=-0.18, y=0.97):
    """Place a bold panel label outside the axes, left of the y-tick labels."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=DARK_SLATE,
    )
