"""Narrow legend column for single-column figures: one box as tall as the panels; each entry is its line sample above
its text, entries separated by dashed lines. entries = [(label, kwargs for ax.plot | None, kwargs for a Rectangle | None)]."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def stacked_legend(fig, x0, y0, x1, y1, entries, fontsize=7.5, frame_color="black", frame_lw=0.8):
    ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(frame_color); sp.set_linewidth(frame_lw)
    n = len(entries); h = 1 / n
    for i, (label, line, patch) in enumerate(entries):
        top = 1 - i * h; ys, yt = top - 0.31 * h, top - 0.68 * h
        if line is not None:
            mk = {k: line[k] for k in ("marker", "ms", "mfc", "mec", "mew") if k in line}
            ax.plot([0.12, 0.88], [ys, ys], **{k: v for k, v in line.items() if k not in mk})
            if mk:
                ax.plot([0.5], [ys], ls="none", **mk)
        if patch is not None:
            ax.add_patch(Rectangle((0.12, ys - 0.13 * h), 0.76, 0.26 * h, lw=0, **patch))
        ax.text(0.5, yt, label, ha="center", va="center", fontsize=fontsize)
        if i < n - 1:
            ax.axhline(top - h, color="black", lw=0.5, ls=(0, (3, 2)))
    return ax
