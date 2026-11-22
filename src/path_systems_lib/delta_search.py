"""Bisection search for delta: the minimal alpha such that path_weight_lp's
check_alpha(G, system, alpha) is feasible.

Consolidates ~9 near-identical bisection loops from metrizable_approximations/,
which had drifted into three different upper-bound conventions (`len(G.nodes())`,
`10.0`, `5.0`) and three different "what if the upper bound isn't feasible either"
strategies (no handling/silently-wrong, a single pre-check, or an iterative
doubling safety net — present, identically, in exactly 2 of the 9 files). This
implementation defaults to the mathematically-justified `len(G.nodes())` bound
(large enough that alpha never needs to exceed it for a chosen-vs-alternative-path
comparison within a connected graph on that many vertices) and always applies the
doubling safety net on top, so it is robust regardless of which convention the
caller is used to.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .path_weight_lp import build_alpha_model

Pair = Tuple[int, int]


@dataclass
class DeltaResult:
    delta: float
    weights: Optional[Dict[Tuple[int, int], float]] = None
    iis: Optional[List[str]] = None


def find_delta(
    G: nx.Graph,
    defined_paths: Dict[Pair, List[int]],
    *,
    lo: float = 1.0,
    hi: Optional[float] = None,
    tol: float = 1e-4,
    eps: float = 1e-4,
    path_cutoff: Optional[int] = None,
    weight_ub: Optional[float] = None,
    auto_expand_hi: bool = True,
    max_hi: float = 1e6,
    compute_iis_at_result: bool = True,
) -> DeltaResult:
    """Binary-search the smallest alpha ("delta") for which `defined_paths` admits
    an edge weighting with every chosen path <= alpha * every alternative path.

    If `hi` isn't feasible, doubles it (up to `max_hi`) before bisecting, so a
    too-small initial guess never silently produces a wrong answer.  Returns
    delta=inf (no weights/iis) if even `max_hi` isn't enough. When the system
    turns out already-strongly-metrizable (delta == 1 within tol), no IIS is
    computed even if compute_iis_at_result is True, since there is no contradiction
    to report.
    """
    if hi is None:
        hi = float(len(G.nodes()))

    def feasible_at(alpha: float):
        return build_alpha_model(G, defined_paths, alpha=alpha, eps=eps, path_cutoff=path_cutoff, weight_ub=weight_ub)

    if auto_expand_hi:
        while not feasible_at(hi).feasible:
            hi *= 2
            if hi > max_hi:
                return DeltaResult(delta=float("inf"))

    best_weights = None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        result = feasible_at(mid)
        if result.feasible:
            hi = mid
            best_weights = result.weights
        else:
            lo = mid

    iis = None
    if compute_iis_at_result and lo > 1.0 + tol:
        iis = build_alpha_model(
            G, defined_paths, alpha=lo, eps=eps, path_cutoff=path_cutoff, weight_ub=weight_ub, compute_iis=True
        ).iis

    return DeltaResult(delta=hi, weights=best_weights, iis=iis)
