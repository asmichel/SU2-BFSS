"""Paper figures from regions.json (plots never call solvers).

Each level has a fixed shade of an ordinal blue ramp.  One run: the shaded allowed region
with a crisp boundary; several runs (increasing level): nested regions.
The black point is the paper's 66-seed variational state (variational/results/seeded_measurement.json
in the companion repository), <R^2> = 8.07419, <u> = 17.11246, <R^4> = 68.7529; its
enclosures are far below the drawing accuracy, so it carries no error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

STATE = {"R2": 8.0741928, "u": 17.1124560, "R4": 68.7529095}   # the paper's variational state
INK, FILL, EDGE = "#0b0b0b", "#b7d3f6", "#1c5cab"
RAMP = ["#0e3e78", "#1c5cab", "#5598e7", FILL]  # darkest for the lowest; L6/L8/L10 match the paper


def shade(level: int) -> str:
    """The fixed shade of a level (levels 4, 6, 8, 10 and up), the same in every figure."""
    return RAMP[min(len(RAMP) - 1, max(0, (level - 4) // 2))]
LABEL = {"R2": r"$\langle R^2\rangle$", "u": r"$\langle u\rangle$", "R4": r"$\langle R^4\rangle$"}
TICKS = {"R2": (1, 0.25), "u": (5, 1), "R4": (20, 5)}
# insets around the state, in the lower right corner of a panel: half-widths of the zoom, major tick spacings,
# and the first level whose region is narrow enough there to need one
INSET = [0.62, 0.11, 0.34, 0.29]
INSETS = {"R2_R4": ((0.1, 2.0), (0.1, 2), 8), "R4_u": ((3.0, 0.6), (2, 0.5), 10)}
# insets labelled from inside, where the zoom sits on a shaded region: shift of the zoom window in X, so that
# the labels clear the data, and the X ticks
INSIDE = {"R4_u": (-1.5, [66, 69])}

# the paper's figure style: STIX text and math fonts, no LaTeX
mpl.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.7, "axes.labelcolor": "#202b35", "text.color": "#202b35",
    "xtick.color": "#202b35", "ytick.color": "#202b35", "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def load(run: Path, planes) -> dict:
    """Read the regions of a run; refuse if any of the planes has not converged."""
    doc = json.loads((run / "regions.json").read_text())
    bad = [pl for pl in planes if not doc["regions"].get(pl, {}).get("ok")]
    if bad:
        raise SystemExit(f"{run}: inner and outer regions disagree in {bad}; no figure")
    return {"level": json.loads((run / "manifest.json").read_text())["level"], **doc["regions"]}


def smooth(poly: np.ndarray, window, iters: int = 3, tolerance: float = 1e-3) -> np.ndarray:
    """Chaikin corner cutting, window vertices pinned.  The cut curve lies inside the outer polygon;
    if it moves away from it by more than `tolerance` (window units), the polygon is drawn as is."""
    (x0, x1), (y0, y1) = window
    span = np.array([x1 - x0, y1 - y0])
    tol = 1e-9 * span.max()
    out, pinned = poly, [any(abs(v[i] - c) < tol for i, c in ((0, x0), (0, x1), (1, y0), (1, y1))) for v in poly]
    for _ in range(iters):
        new, pin = [], []
        for i, v in enumerate(out):
            if pinned[i]:
                new.append(v)
                pin.append(True)
            else:
                new += [0.75 * v + 0.25 * out[i - 1], 0.75 * v + 0.25 * out[(i + 1) % len(out)]]
                pin += [False, False]
        out, pinned = np.array(new), pin
    a, b = out / span, np.roll(out, -1, axis=0) / span        # distance of each outer vertex to the cut curve
    ab = b - a
    worst = max(np.linalg.norm(v - a - np.clip(((v - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-300), 0, 1)[:, None]
                               * ab, axis=1).min() for v in poly / span)
    return out if worst <= tolerance else poly


def bounds(poly: np.ndarray, window) -> np.ndarray:
    """The polygon's edges that are bounds, not window clipping (NaN-separated x, y)."""
    (x0, x1), (y0, y1) = window
    tol = 1e-9 * max(x1 - x0, y1 - y0)
    out = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        if not any(abs(p[i] - c) < tol and abs(q[i] - c) < tol for i, c in ((0, x0), (0, x1), (1, y0), (1, y1))):
            out += [p, q, (np.nan, np.nan)]
    return np.array(out).T if out else np.zeros((2, 0))


def panel(ax, regs: list[dict], plane: str, inset: bool) -> None:
    """Draw one plane: shaded regions, their bounding edges, the state, and the inset."""
    g0 = regs[0][plane]
    X, Y = g0["X"], g0["Y"]
    (x0, x1), (y0, y1) = g0["window"]
    single = len(regs) == 1
    for k, reg in enumerate(regs):
        poly = smooth(np.array(reg[plane]["outer"]), g0["window"])
        ax.add_patch(Polygon(poly, closed=True, facecolor=shade(reg["level"]), edgecolor="none", zorder=2 + k))
        ax.plot(*bounds(poly, g0["window"]), color=EDGE if single else "white", linewidth=0.9 if single else 0.55,
                solid_joinstyle="round", solid_capstyle="round", zorder=2 + k)
    _state(ax, X, Y)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(LABEL[X], labelpad=2)
    ax.set_ylabel(LABEL[Y], labelpad=2)
    for axis, name in ((ax.xaxis, X), (ax.yaxis, Y)):
        axis.set_major_locator(MultipleLocator(TICKS[name][0]))
        axis.set_minor_locator(MultipleLocator(TICKS[name][1]))
    if inset and plane in INSETS and regs[-1]["level"] >= INSETS[plane][2]:
        (zx, zy), (tx, ty), _ = INSETS[plane]
        cx = STATE[X] + (INSIDE[plane][0] if plane in INSIDE else 0)
        ins = ax.inset_axes(INSET)
        for reg in regs:
            poly = smooth(np.array(reg[plane]["outer"]), g0["window"])
            ins.add_patch(Polygon(poly, closed=True, edgecolor="none",
                                  facecolor=shade(reg["level"])))
            ins.plot(*bounds(poly, g0["window"]), color=EDGE if single else "white", linewidth=0.8)
        _state(ins, X, Y)
        ins.set_xlim(cx - zx, cx + zx)
        ins.set_ylim(STATE[Y] - zy, STATE[Y] + zy)
        ins.xaxis.set_major_locator(MultipleLocator(tx))
        ins.yaxis.set_major_locator(MultipleLocator(ty))
        ins.tick_params(labelsize=6.5, which="both", length=2, width=0.5)
        ins.minorticks_off()
        if plane in INSIDE:
            ins.set_xticks(INSIDE[plane][1])
            ins.tick_params(direction="in")
            ins.tick_params(axis="x", pad=-2)
            ins.tick_params(axis="y", pad=-2)
            for label in ins.get_yticklabels():
                label.set_horizontalalignment("left")
            for label in ins.get_xticklabels():
                label.set_verticalalignment("bottom")
        ax.add_patch(Rectangle((cx - zx, STATE[Y] - zy), 2 * zx, 2 * zy, fill=False,
                               edgecolor=INK, linewidth=0.5, zorder=10))


def _state(ax, X: str, Y: str) -> None:
    """The variational state."""
    ax.plot(STATE[X], STATE[Y], "o", markersize=3.2, color=INK, markeredgewidth=0, zorder=9)


def figure(runs: list[Path], out: Path, planes=("R2_u", "R2_R4", "R4_u"), tag: str = "peninsula") -> None:
    """The combined figure and one figure per plane, for one run or several levels."""
    regs = [load(r, planes) for r in runs]
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_" + "_".join(f"L{r['level']}" for r in regs)

    def legend(fig):
        if len(regs) > 1:
            hs = [Patch(facecolor=shade(r["level"]), label=f"level {r['level']}") for r in regs]
            fig.legend(handles=hs, loc="outside upper center", ncol=len(hs), frameon=False)

    fig, axes = plt.subplots(1, len(planes), figsize=(7.0 * len(planes) / 3, 2.35 if len(regs) == 1 else 2.55),
                             constrained_layout=True)
    for ax, pl in zip(axes, planes):
        panel(ax, regs, pl, inset=True)
    legend(fig)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_combined.{ext}")
    plt.close(fig)
    for pl in planes:
        fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
        panel(ax, regs, pl, inset=True)
        legend(fig)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{stem}_{pl}.{ext}")
        plt.close(fig)
