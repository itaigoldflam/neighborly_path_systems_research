import gurobipy as gp
import networkx as nx
from itertools import combinations

# This file is intended to check whether there exist path systems 
# on wheel graphs that satisfy all triangle inequalities but are 
# not neighborly.

# Small epsilon for strict inequalities when requested
EPSILON = 1e-6


def ts(*args):
    """Return the unordered pair as a sorted tuple."""
    if len(args) == 1:
        return tuple(sorted(args[0]))
    if len(args) == 2:
        return tuple(sorted((args[0], args[1])))


def _add_wheel_triangle_inequalities(model, d_vars, n, strict=True):
    """
    Add triangle inequalities for all wheel triangles (0,a,a+1)
    where 0 is the hub and a ranges over rim vertices.
    """
    hub = 0
    vtxs = list(range(n))
    rim = [v for v in vtxs if v != hub]
    rim.sort()
    m = len(rim)
    epsilon = EPSILON if strict else 0.0
    for idx, a in enumerate(rim):
        b = rim[(idx + 1) % m]
        i0a = ts(hub, a)
        i0b = ts(hub, b)
        iab = ts(a, b)
        model.addConstr(d_vars[i0a] <= d_vars[i0b] + d_vars[iab] - epsilon)
        model.addConstr(d_vars[iab] <= d_vars[i0a] + d_vars[i0b] - epsilon)
        model.addConstr(d_vars[i0b] <= d_vars[i0a] + d_vars[iab] - epsilon)


def _add_wheel_outer_cycle_inequalities(model, d_vars, n, strict=True):
    """
    Add inequalities for the outer rim cycle (1,2,...,n-1):
    every rim edge length is strictly smaller than the sum of
    the lengths of the remaining rim edges.
    """
    hub = 0
    rim = [v for v in range(n) if v != hub]
    rim.sort()
    epsilon = EPSILON if strict else 0.0
    cycle_edges = [
        ts(rim[i], rim[(i + 1) % len(rim)])
        for i in range(len(rim))
    ]
    for edge in cycle_edges:
        others = [d_vars[e] for e in cycle_edges if e != edge]
        model.addConstr(d_vars[edge] <= gp.quicksum(others) - epsilon)


def _wheel_lp_feasible_with_cycle_edge_inequality(n, cycle, edge_pair, strict=True):
    """
    Build an LP with:
      - variables d(i,j) for all unordered vertex pairs
      - all wheel triangle inequalities (0,a,a+1)
      - all outer rim inequalities
      - one extra inequality: on the given cycle, the length of
        edge_pair is at least the sum of the other edges on the cycle.
    Return (True, distances) iff the LP is feasible, otherwise (False, None),
    where distances[(a,b)] = d(a,b) for all unordered pairs {a,b}.
    """
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            vtxs = list(range(n))
            d = {}
            for a, b in combinations(vtxs, 2):
                i, j = ts(a, b)
                d[(i, j)] = model.addVar(
                    name=f"d({i},{j})",
                    vtype=gp.GRB.CONTINUOUS,
                    obj=0,
                    lb=1.0,
                )

            # add triangle inequalities and outer rim inequalities
            _add_wheel_triangle_inequalities(model, d, n, strict=strict)
            _add_wheel_outer_cycle_inequalities(model, d, n, strict=strict)

            # prepare the cycle edges as unordered pairs
            k = len(cycle)
            cycle_edges = []
            for idx in range(k):
                u = cycle[idx]
                v = cycle[(idx + 1) % k]
                cycle_edges.append(ts(u, v))

            edge_pair = ts(*edge_pair)
            if edge_pair not in cycle_edges:
                raise ValueError("edge_pair must be an edge of the given cycle")

            epsilon = EPSILON if strict else 0.0
            lhs = d[edge_pair]
            rhs_terms = [d[e] for e in cycle_edges if e != edge_pair]
            model.addConstr(lhs >= gp.quicksum(rhs_terms) + epsilon)

            model.optimize()

            if model.Status == gp.GRB.OPTIMAL:
                distances = {}
                for a, b in combinations(vtxs, 2):
                    i, j = ts(a, b)
                    distances[(a, b)] = d[(i, j)].X
                return True, distances
            else:
                return False, None


def is_wheel_neighborly_via_lp(n, verbose=False, strict=True):
    """
    Check neighborliness of the n-wheel using the LP procedure:
      1. For each simple cycle in the wheel (length > 3).
      2. For each edge on that cycle.
      3. Build an LP with all triangle inequalities for (0,a,a+1),
         the outer rim inequalities, and the opposite inequality
         'this edge >= sum(other edges on the cycle)'.
      4. Require INfeasibility for every such LP; if any is feasible,
         the triangle inequalities do not imply the original inequality.
    Return True iff all these LPs are infeasible.
    """
    G = nx.wheel_graph(n)
    cycles = nx.simple_cycles(G)
    for cycle in cycles:
        k = len(cycle)
        if k <= 3:
            continue
        edges = [ts(cycle[i], cycle[(i + 1) % k]) for i in range(k)]
        for edge in edges:
            feasible, distances = _wheel_lp_feasible_with_cycle_edge_inequality(
                n, cycle, edge, strict=strict
            )
            if verbose:
                print(f"Cycle {cycle}, edge {edge}, feasible = {feasible}")
            if feasible:
                print(
                    f"Counterexample found for n={n}, cycle={cycle}, edge={edge} "
                    f"({'strict' if strict else 'non-strict'} inequalities):"
                )
                for (a, b), w in sorted(distances.items()):
                    print(f"d({a},{b}) = {w}")
                return False
    return True


def check_wheel_neighborly_lp(n, strict=True):
    """
    Convenience wrapper that prints whether the n-wheel is neighborly
    according to the LP-based triangle/cycle inequality test.
    """
    is_neighborly = is_wheel_neighborly_via_lp(n, strict=strict)
    print(
        f"The {n}-wheel IS neighborly (LP test, {'strict' if strict else 'non-strict'})."
        if is_neighborly
        else f"The {n}-wheel IS NOT neighborly (LP test, {'strict' if strict else 'non-strict'})."
    )


if __name__ == "__main__":
    for n in range(3, 8):
        check_wheel_neighborly_lp(n)

