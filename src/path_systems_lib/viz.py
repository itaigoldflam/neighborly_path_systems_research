"""Small plotting helpers for visualizing path systems on 2D grid graphs.

Consolidates `get_maximal_paths`/`draw_grid_system`, which were identical copies in
metrizable_grid_on_torus/check_m_n_grid_metrizability_print_graph.py and its
_weak.py sibling.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

Pair = Tuple[int, int]
GridVertex = Tuple[int, int]


def get_maximal_paths(paths: Dict[Pair, List[GridVertex]]) -> List[List[GridVertex]]:
    """From a path system's dict of chosen paths, return only the ones that are
    not a sub-path of any other (longer) chosen path — the "interesting" ones to
    draw, since sub-paths are implied by their containing path."""
    unique_paths = []
    seen = set()
    for p in paths.values():
        if len(p) > 2:
            p_tuple, p_rev_tuple = tuple(p), tuple(p[::-1])
            if p_tuple not in seen and p_rev_tuple not in seen:
                unique_paths.append(p)
                seen.add(p_tuple)

    maximal_paths = []
    for i, p1 in enumerate(unique_paths):
        is_maximal = True
        for j, p2 in enumerate(unique_paths):
            if i == j:
                continue
            str_p1 = str(p1).strip("[]")
            str_p2 = str(p2).strip("[]")
            str_p2_rev = str(p2[::-1]).strip("[]")
            if str_p1 in str_p2 or str_p1 in str_p2_rev:
                is_maximal = False
                break
        if is_maximal:
            maximal_paths.append(p1)

    return maximal_paths


def draw_grid_system(
    rows: int, cols: int, maximal_paths: List[List[GridVertex]], filename: Union[str, Path]
) -> None:
    """Draw a rows x cols grid with the given paths overlaid (slightly offset per
    path so overlapping segments stay visually distinguishable), saved to `filename`."""
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))

    for i in range(rows):
        for j in range(cols):
            ax.plot(j, -i, "ko", markersize=8, color="lightgray")
            if j + 1 < cols:
                ax.plot([j, j + 1], [-i, -i], color="whitesmoke", linewidth=2, zorder=0)
            if i + 1 < rows:
                ax.plot([j, j], [-i, -(i + 1)], color="whitesmoke", linewidth=2, zorder=0)

    for i in range(rows):
        for j in range(cols):
            ax.text(j, -i + 0.15, f"({i},{j})", ha="center", va="bottom", fontsize=8, color="gray")

    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(maximal_paths) > len(colors):
        colors = plt.cm.get_cmap("hsv", len(maximal_paths))(np.linspace(0, 1, len(maximal_paths)))

    for path_idx, path in enumerate(maximal_paths):
        color = colors[path_idx % len(colors)]
        offset_val = (path_idx % 5 - 2) * 0.05
        path_x = [v[1] + offset_val for v in path]
        path_y = [-v[0] + offset_val for v in path]
        ax.plot(path_x, path_y, color=color, linewidth=3, alpha=0.8, marker="o", markersize=5, label=f"Path {path_idx + 1}")

    ax.set_aspect("equal")
    ax.axis("off")
    if maximal_paths:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
