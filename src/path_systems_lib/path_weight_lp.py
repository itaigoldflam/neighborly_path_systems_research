"""Direct path-weight-sum LP: does an edge weighting exist under which every
chosen path is at most `alpha` times as long as every alternative path between
its endpoints?

Consolidates two LP formulations that had drifted into "Family A" (WEAK/STRONG/
ALPHA modes, no epsilon in ALPHA mode, optional weight upper bound in STRONG mode
— metrizable_approximations/find_delta_for_paley.py and 2 siblings) and "Family B"
(a single always-epsilon'd alpha constraint, used even to test alpha=1 — 7 other
metrizable_approximations files) across 9 near-duplicate implementations, into one
`build_alpha_model` with `alpha`/`eps`/`weight_ub` as explicit parameters instead of
a hidden "mode" string. WEAK is alpha=1, eps=0; STRONG is alpha=1, eps=strong_eps;
a delta/alpha search (see delta_search.py) sweeps alpha with a small numerical eps.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from .gurobi_utils import quiet_model

Edge = Tuple[int, int]
Pair = Tuple[int, int]


@dataclass
class AlphaModelResult:
    feasible: bool
    weights: Optional[Dict[Edge, float]] = None
    iis: Optional[List[str]] = None


def _path_weight_expr(path, w: Dict[Edge, gp.Var]) -> gp.LinExpr:
    expr = gp.LinExpr()
    for i in range(len(path) - 1):
        edge = tuple(sorted((path[i], path[i + 1])))
        expr += w[edge]
    return expr


def build_alpha_model(
    G: nx.Graph,
    defined_paths: Dict[Pair, List[int]],
    alpha: float = 1.0,
    *,
    eps: float = 0.0,
    weight_ub: Optional[float] = None,
    path_cutoff: Optional[int] = None,
    skip_trivial_edges: bool = False,
    objective: Optional[str] = None,
    compute_iis: bool = False,
) -> AlphaModelResult:
    """Test whether edge weights w (each >= 1, optionally <= weight_ub) exist such
    that for every chosen pair (u, v) -> path in `defined_paths`:

        weight(chosen_path) <= alpha * weight(alt_path) - eps

    for every alternative simple u-v path in G (up to `path_cutoff` edges, if set).
    Pass alpha=1, eps=0 for WEAK metrizability; alpha=1, eps=<small> for STRONG;
    sweep alpha via delta_search.find_delta for the minimal-alpha ("delta") question.

    skip_trivial_edges=True skips constraining pairs whose chosen path is a direct
    graph edge (length 2) — matches one specific original script
    (find_delta_for_paley.py) that opted out of this for its own reasons; every
    other original file (and the default here) DOES constrain direct edges against
    longer alternatives, since an edge can genuinely be "shortcut-forced" short by a
    competing longer path (e.g. a theta graph's rim edges vs. the long way around).

    objective="sum" minimizes total edge weight instead of just testing feasibility
    (used when the caller wants the actual minimal weighting, not just a yes/no).
    """
    model = quiet_model("PathSystem_Alpha")
    w: Dict[Edge, gp.Var] = {}
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        kwargs = {"lb": 1.0, "vtype": GRB.CONTINUOUS, "name": f"w_{edge[0]}_{edge[1]}"}
        if weight_ub is not None:
            kwargs["ub"] = weight_ub
        w[edge] = model.addVar(**kwargs)

    constr_descriptions: Dict[gp.Constr, str] = {}
    for (u, v), chosen_path in defined_paths.items():
        if u == v or (skip_trivial_edges and len(chosen_path) <= 2):
            continue
        # Normalize to tuple: callers may store paths as tuples (path_systems.py's
        # generators) or lists (nx.all_simple_paths, and some original scripts'
        # hand-rolled backtrackers) — comparing across those types with `==` is
        # always False in Python even for identical contents, which would silently
        # let a chosen path be compared against itself as if it were a genuine
        # alternative (producing an unsatisfiable "chosen < chosen" constraint).
        chosen_path_t = tuple(chosen_path)
        chosen_expr = _path_weight_expr(chosen_path, w)
        for alt_path in nx.all_simple_paths(G, u, v, cutoff=path_cutoff):
            alt_path_t = tuple(alt_path)
            if alt_path_t == chosen_path_t or alt_path_t == chosen_path_t[::-1]:
                continue
            alt_expr = _path_weight_expr(alt_path, w)
            c = model.addConstr(chosen_expr <= alpha * alt_expr - eps)
            constr_descriptions[c] = (
                f"weight({chosen_path}) <= {alpha} * weight({alt_path}) - {eps}"
            )

    if objective == "sum":
        model.setObjective(gp.quicksum(w.values()), GRB.MINIMIZE)
    elif objective is not None:
        raise ValueError("objective must be 'sum' or None")

    model.optimize()
    if model.Status == GRB.OPTIMAL:
        weights = {edge: var.X for edge, var in w.items()}
        return AlphaModelResult(feasible=True, weights=weights)

    iis = None
    if compute_iis and model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        iis = [constr_descriptions.get(c, c.ConstrName) for c in model.getConstrs() if c.IISConstr]
    return AlphaModelResult(feasible=False, iis=iis)


def check_weak(G: nx.Graph, defined_paths, **kwargs) -> AlphaModelResult:
    """Is there a weighting making every chosen path <= every alternative (non-strict)?"""
    return build_alpha_model(G, defined_paths, alpha=1.0, eps=0.0, **kwargs)


def check_strong(G: nx.Graph, defined_paths, *, eps: float = 1e-3, weight_ub: Optional[float] = 1000.0, **kwargs) -> AlphaModelResult:
    """Is there a weighting making every chosen path strictly shorter than every
    alternative? weight_ub bounds weights for numerical stability, matching the
    original STRONG-mode behavior."""
    return build_alpha_model(G, defined_paths, alpha=1.0, eps=eps, weight_ub=weight_ub, **kwargs)


def check_alpha(G: nx.Graph, defined_paths, alpha: float, *, eps: float = 1e-4, **kwargs) -> AlphaModelResult:
    """Is there a weighting making every chosen path <= alpha * every alternative
    (with a small numerical safety margin eps)? The building block for delta_search."""
    return build_alpha_model(G, defined_paths, alpha=alpha, eps=eps, **kwargs)
