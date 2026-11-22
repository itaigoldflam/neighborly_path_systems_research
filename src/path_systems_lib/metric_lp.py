"""Distance-variable metric-feasibility LP.

Given a path system, tests whether a consistent metric exists: distances d(a,b)
that agree with path-sum along every chosen path, and satisfy the triangle
inequality (strictly or weakly) off-path. Ported from
metrizable_wheels/utils/Metric_Tests.py, which was already the most general
implementation of this formulation in the repo (the other ~8 call sites of this
formulation duplicated its LP construction inline rather than importing it).
"""
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np

from .gurobi_utils import quiet_env
from .r_l_vectors import ts

Pair = Tuple[int, int]


def get_vtxs_from_system(pSystem) -> list:
    vtxs = set()
    for pair in pSystem:
        vtxs |= set(pSystem[pair])
    return sorted(vtxs)


def get_vtxs_from_triple_system(t_system) -> list:
    vtxs = set()
    for (a, b), c in t_system:
        vtxs |= {a, b, c}
    return sorted(vtxs)


def get_graph_from_system(pSystem) -> nx.Graph:
    """Reconstruct the underlying graph from a path system's edges."""
    G = nx.Graph()
    for (u, v) in pSystem:
        path = pSystem[(u, v)]
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])
    return G


def build_metric_model(
    pSystem, *, strict: bool = True, strict_gap: float = 1.0, env: gp.Env, describe_constraints: bool = False
) -> Tuple[gp.Model, Dict[Pair, gp.Var], Optional[Dict[gp.Constr, str]]]:
    """Build (but don't solve) the metric-feasibility LP for `pSystem`:

        d(a,c) + d(c,b) - d(a,b) = 0            if c lies on the chosen path P(a,b)
        d(a,b) + strict_gap*s <= d(a,c) + d(c,b) if c does not lie on P(a,b)  (s=1 iff strict)
        d(a,b) >= 1                              for every pair (a,b)

    `strict_gap` controls how large a slack "strict" enforces: the default of 1.0
    matches `is_system_metric`'s original behavior (distances are meant to be
    meaningfully separated by whole units); pass a small epsilon instead when the
    model will additionally be given a sum/max objective to minimize (a unit gap
    would otherwise dominate and distort the optimum) — see `optimal_metric_value`.

    Set describe_constraints=True to also get back a {constraint: human-readable
    description} dict, useful for rendering an IIS (see `check_metric`).

    Returns (model, d, descriptions) with d keyed by (min(a,b), max(a,b));
    descriptions is None unless describe_constraints=True. Caller is responsible
    for calling model.optimize().
    """
    model = gp.Model(env=env)
    vtxs = get_vtxs_from_system(pSystem)
    d = {ts(a, b): model.addVar(name=f"d({a},{b})", vtype=gp.GRB.CONTINUOUS, lb=1) for a, b in combinations(vtxs, 2)}
    sv = model.addVar(name="strict", vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
    sv_constr = model.addConstr(sv == (1 if strict else 0))

    descriptions: Optional[Dict[gp.Constr, str]] = None
    if describe_constraints:
        descriptions = {sv_constr: f"strict = {1 if strict else 0}"}

    for a, b in combinations(vtxs, 2):
        for c in vtxs:
            if c == a or c == b:
                continue
            if (a, b) in pSystem and c in pSystem[(a, b)]:
                constr = model.addConstr(d[ts(a, b)] - d[ts(a, c)] - d[ts(c, b)] == 0)
                if descriptions is not None:
                    descriptions[constr] = f"d({a},{b}) = d({a},{c}) + d({c},{b})   [{c} lies on chosen path]"
            elif (a, b) in pSystem:
                constr = model.addConstr(d[ts(a, b)] - d[ts(a, c)] - d[ts(c, b)] + strict_gap * sv <= 0)
                if descriptions is not None:
                    descriptions[constr] = f"d({a},{b}) <= d({a},{c}) + d({c},{b})   [alternative via {c}]"
            else:
                constr = model.addConstr(d[ts(a, b)] - d[ts(a, c)] - d[ts(c, b)] <= 0)
                if descriptions is not None:
                    descriptions[constr] = f"d({a},{b}) <= d({a},{c}) + d({c},{b})"

    return model, d, descriptions


def is_system_metric(pSystem, strict: bool = True) -> Optional[Dict[Pair, float]]:
    """Test whether `pSystem` admits a consistent metric.

    Returns the solved distances keyed by (min(a,b), max(a,b)) if feasible, else None.
    """
    with quiet_env() as env:
        model, d, _ = build_metric_model(pSystem, strict=strict, env=env)
        with model:
            vtxs = get_vtxs_from_system(pSystem)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                return None
            return {(i, j): d[(i, j)].X for i, j in combinations(vtxs, 2)}


@dataclass
class MetricResult:
    feasible: bool
    distances: Optional[Dict[Pair, float]] = None
    iis: Optional[List[str]] = None
    all_constraints: Optional[List[str]] = None


def check_metric(pSystem, *, strict: bool = True, compute_iis: bool = False) -> MetricResult:
    """Like `is_system_metric`, but optionally also extracts a human-readable IIS
    (the exact subset of contradictory constraints) when infeasible — used by
    callers that need to *explain* non-metrizability, not just detect it."""
    with quiet_env() as env:
        model, d, descriptions = build_metric_model(pSystem, strict=strict, env=env, describe_constraints=compute_iis)
        with model:
            vtxs = get_vtxs_from_system(pSystem)
            model.optimize()
            if model.Status == gp.GRB.OPTIMAL:
                return MetricResult(feasible=True, distances={(i, j): d[(i, j)].X for i, j in combinations(vtxs, 2)})

            iis, all_constraints = None, None
            if compute_iis and model.Status == gp.GRB.INFEASIBLE:
                all_constraints = list(descriptions.values())
                model.computeIIS()
                iis = [descriptions.get(c, c.ConstrName) for c in model.getConstrs() if c.IISConstr]
            return MetricResult(feasible=False, iis=iis, all_constraints=all_constraints)


def optimal_metric_value(
    pSystem,
    edges,
    *,
    objective: str = "sum",
    strict: bool = True,
    strict_gap: float = 1e-6,
) -> Optional[Tuple[float, Dict[Pair, float]]]:
    """Among all metrics consistent with `pSystem`, find the one minimizing either
    the sum or the max of d over `edges` (an iterable of (u, v) pairs — typically
    the graph's actual edges, since only those have a physical "length" meaning).

    Uses a small `strict_gap` by default (unlike `is_system_metric`'s default of a
    full unit) so that "strict" doesn't itself dominate the objective being
    minimized. Returns (optimal_value, distances) or None if infeasible.
    """
    with quiet_env() as env:
        model, d, _ = build_metric_model(pSystem, strict=strict, strict_gap=strict_gap, env=env)
        with model:
            edge_keys = [ts(*e) for e in edges]
            if objective == "sum":
                model.setObjective(gp.quicksum(d[e] for e in edge_keys), gp.GRB.MINIMIZE)
            elif objective == "max":
                m_var = model.addVar(name="M_max_edge", vtype=gp.GRB.CONTINUOUS, lb=1)
                for e in edge_keys:
                    model.addConstr(m_var >= d[e])
                model.setObjective(m_var, gp.GRB.MINIMIZE)
            else:
                raise ValueError("objective must be 'sum' or 'max'")

            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                return None
            return model.ObjVal, {key: var.X for key, var in d.items()}


def _pair_to_ind(i: int, j: int) -> int:
    a, b = ts(i, j)
    return b * (b - 1) // 2 + a


def is_triple_system_metric(t_system, check_closure: bool = False) -> Optional[Dict[Pair, float]]:
    """Same feasibility question as `is_system_metric`, phrased over a collection of
    pointed triples ((a, b), c) meaning "c lies on the a-b path" instead of a full
    path system. Distances are unbounded below (lb=0), and off-path pairs must
    strictly satisfy the triangle inequality (>= 1 slack), matching the original
    check_wheel-family usage of this function.

    check_closure=True additionally reports (via matrix-rank comparison) which
    non-collinear triples are implied by the given ones, printing progress; kept
    for parity with the original but rarely needed.
    """
    with quiet_env() as env:
        with gp.Model(env=env) as model:
            vtxs = get_vtxs_from_triple_system(t_system)
            if check_closure:
                t_system = _close_triple_system(vtxs, t_system)

            d = {ts(a, b): model.addVar(name=f"d({a},{b})", vtype=gp.GRB.CONTINUOUS, lb=0) for a, b in combinations(vtxs, 2)}
            for a, b in combinations(vtxs, 2):
                for c in vtxs:
                    if c == a or c == b:
                        continue
                    if ((a, b), c) in t_system:
                        model.addConstr(d[ts(a, c)] + d[ts(c, b)] - d[ts(a, b)] == 0)
                    else:
                        model.addConstr(d[ts(a, c)] + d[ts(c, b)] - d[ts(a, b)] >= 1)

            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                return None
            return {(i, j): d[(i, j)].X for i, j in combinations(vtxs, 2)}


def _close_triple_system(vtxs, t_system):
    """Add every triple implied (by linear-algebraic closure) by t_system."""
    vtx_to_ind = {v: i for i, v in enumerate(vtxs)}
    n, m = len(vtxs), len(t_system)
    M = np.zeros((m + 1, n * (n - 1) // 2))
    for ind, ((a, b), c) in enumerate(t_system):
        M[ind][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[b])] = -1
        M[ind][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[c])] = 1
        M[ind][_pair_to_ind(vtx_to_ind[c], vtx_to_ind[b])] = 1
    rank = np.linalg.matrix_rank(M)

    to_add = set()
    for a, b in combinations(vtxs, 2):
        for c in vtxs:
            if c == a or c == b:
                continue
            M[m][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[b])] = -1
            M[m][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[c])] = 1
            M[m][_pair_to_ind(vtx_to_ind[c], vtx_to_ind[b])] = 1
            if rank == np.linalg.matrix_rank(M):
                to_add.add(((a, b), c))
            M[m][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[b])] = 0
            M[m][_pair_to_ind(vtx_to_ind[a], vtx_to_ind[c])] = 0
            M[m][_pair_to_ind(vtx_to_ind[c], vtx_to_ind[b])] = 0
    return t_system | to_add


def non_metric_certificate(pSystem, strict: bool = True) -> Optional[Dict[Tuple[int, int, int], float]]:
    """When `pSystem` is not metric, find a witness: a nonnegative combination of
    triangle-inequality/path-equality constraints that sums to a contradiction
    (0 < 0). This solves the dual of the LP in `is_system_metric` and checks it is
    unbounded/feasible-with-positive-objective. Returns the combination's
    coefficients keyed by (i, j, k), or None if the system is in fact metric.
    """
    vtxs = sorted(get_vtxs_from_system(pSystem))
    model = gp.Model()
    d, e, strict_constraint = {}, {}, []

    for a, b in combinations(vtxs, 2):
        e[(a, b)] = model.addVar(name=f"({a},{b})", vtype=gp.GRB.CONTINUOUS, obj=0, lb=0, ub=float("inf"))

    for a, b, c in combinations(vtxs, 3):
        for i, j in combinations([a, b, c], 2):
            k = next(x for x in (a, b, c) if x != i and x != j)
            if (i, j) in pSystem and k in pSystem[(i, j)]:
                d[(i, j, k)] = model.addVar(name=f"({i},{j},{k})", vtype=gp.GRB.CONTINUOUS, obj=0, lb=-float("inf"), ub=float("inf"))
            elif (i, j) in pSystem:
                d[(i, j, k)] = model.addVar(name=f"({i},{j},{k})", vtype=gp.GRB.CONTINUOUS, obj=0, lb=0, ub=float("inf"))
                strict_constraint.append((i, j, k))
            else:
                d[(i, j, k)] = model.addVar(name=f"({i},{j},{k})", vtype=gp.GRB.CONTINUOUS, obj=0, lb=0, ub=float("inf"))

    for a, b in combinations(vtxs, 2):
        model.addConstr(
            gp.quicksum(d[(a, b, c)] for c in vtxs if c != a and c != b)
            + gp.quicksum(-d[(*ts(a, c), b)] for c in vtxs if c != a and c != b)
            + gp.quicksum(-d[(*ts(b, c), a)] for c in vtxs if c != a and c != b)
            - e[(a, b)] >= 0
        )

    if strict:
        model.addConstr(gp.quicksum(d[t] for t in strict_constraint) + gp.quicksum(e[s] for s in combinations(vtxs, 2)) >= 1)
    else:
        model.addConstr(gp.quicksum(e[(a, b)] for a, b in permutations(vtxs, 2)) >= 1)

    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        return None
    return {(i, j, k): d[(i, j, k)].X for a, b, c in combinations(vtxs, 3) for i, j in combinations([a, b, c], 2) for k in [next(x for x in (a, b, c) if x != i and x != j)]}


def format_non_metric_certificate(vtxs, coeff: Dict[Tuple[int, int, int], float]) -> str:
    """Render a non_metric_certificate() result as human-readable triangle inequalities."""
    pos_lines, neg_lines = [], []
    for a, b, c in combinations(vtxs, 3):
        for i, j in combinations([a, b, c], 2):
            k = next(x for x in (a, b, c) if x != i and x != j)
            x1, x2 = ts(i, k)
            y1, y2 = ts(j, k)
            coefficient = coeff[(i, j, k)]
            if coefficient > 0:
                body = f"d({i},{j}) <= d({x1},{x2}) + d({y1},{y2})"
                pos_lines.append(body if coefficient == 1 else f"{coefficient}[{body}]")
            elif coefficient < 0:
                body = f"d({x1},{x2}) + d({y1},{y2}) <= d({i},{j})"
                neg_lines.append(body if coefficient == -1 else f"{-coefficient}[{body}]")
    return "\n".join(pos_lines) + "\n\n" + "\n".join(neg_lines)
