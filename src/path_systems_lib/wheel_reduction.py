"""Wheel-reduction pipeline: repeatedly contract a reducible rim edge of a wheel's
R-vector down to a smaller wheel, accumulating linear inequalities relating the
deleted edges' lengths (t_k) to the final regular wheel's vertex weights (h_v),
then test metrizability by solving the resulting LP.

Consolidates the Reduction_to_regular/ family (reduction.py, reduction_backtracking.py,
reduction_debug.py, infeasible_hunter.py), whose core primitives
(`get_reducible_edges*`, `reduce_wheel`, `pi`, `solve_system`) were byte-identical
across all four files modulo docstrings/comments. `wheel_geometry/reduction.py` is a
genuinely different contraction algorithm (different merge formula, edge-selection
heuristic, and termination rule) and is deliberately NOT folded in here — see
research_log.md for the write-up asking which one is intended to be canonical.
"""
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .gurobi_utils import extract_iis
from .r_l_vectors import get_L_from_R

Direction = str  # 'R' or 'L'


def get_reducible_edges_right(R: List[int]) -> List[int]:
    """Rim edges (i, i+1) that no chosen path currently ends at — safe to
    contract without invalidating an already-placed path endpoint."""
    n = len(R)
    endpoints = {(i + R[i]) % n for i in range(n)}
    return [i for i in range(n) if i not in endpoints]


def get_reducible_edges_left(L: List[int]) -> List[int]:
    """Mirror of get_reducible_edges_right using L (counter-clockwise reach):
    rim edges (i, i+1) that no chosen path currently starts at."""
    n = len(L)
    startpoints = {(i - L[i]) % n for i in range(n)}
    return [i for i in range(n) if (i + 1) % n not in startpoints]


def reduce_wheel(R: List[int], i: int) -> List[int]:
    """Contract rim edge (i, i+1), merging their reach into one surviving vertex
    and decrementing every other vertex's reach that spanned across the
    contracted edge."""
    n = len(R)
    R_red = [0] * (n - 1)
    for v in range(n):
        dist_v_to_i = (i - v) % n
        adjusted_reach = R[v] - 1 if dist_v_to_i < R[v] else R[v]
        mapped_v = v % (n - 1) if v <= i else (v - 1) % (n - 1)
        if v == i or v == (i + 1) % n:
            R_red[mapped_v] = max(R_red[mapped_v], adjusted_reach)
        else:
            R_red[mapped_v] = adjusted_reach
    return R_red


def pi(x: int, i: int, n: int) -> int:
    """Project rim index x through the contraction of edge i in an n-cycle onto
    the resulting (n-1)-cycle's indexing."""
    return x % (n - 1) if x <= i else (x - 1) % (n - 1)


def find_reduction_wheels(R: List[int]) -> Tuple[List[List[int]], List[int], List[list], List[list]]:
    """Greedily reduce R by always contracting the first right-reducible edge,
    until none remain. Returns (W_list, indexes_reduced, big_deltas_list,
    small_deltas_list) — the single deterministic reduction path used by the
    original (non-backtracking) reduction pipeline."""
    reduction_path = []
    current_R = R.copy()
    while True:
        reducible = get_reducible_edges_right(current_R)
        if not reducible:
            break
        i = reducible[0]
        reduction_path.append((i, "R"))
        current_R = reduce_wheel(current_R, i)
    return build_system_from_path(R, reduction_path)[:4]


def build_system_from_path(
    R_initial: List[int], reduction_path: List[Tuple[int, Direction]]
) -> Tuple[List[List[int]], List[int], List[list], List[list], List[Direction]]:
    """Apply an explicit sequence of (edge_index, 'R'|'L') reductions to R_initial,
    accumulating the big/small-delta inequality data at each step. Returns
    (W_list, indexes_reduced, big_deltas_list, small_deltas_list, directions_used).
    """
    W_list = [R_initial]
    indexes_reduced: List[int] = []
    big_deltas_list: List[list] = []
    small_deltas_list: List[list] = []
    directions_used: List[Direction] = []
    current_R = R_initial.copy()

    for i, direction in reduction_path:
        n = len(current_R)
        indexes_reduced.append(i)
        directions_used.append(direction)

        old_R = W_list[-1]
        old_L = get_L_from_R(old_R)
        big_delta_paths = []
        small_delta_paths = []

        if direction == "R":
            for j in range(old_L[i], -1, -1):
                start_ind = (i - j) % n
                prev_ind = (start_ind - 1) % n
                if old_R[start_ind] > j and old_R[start_ind] > old_R[prev_ind] - 1:
                    end_ind = (start_ind + old_R[start_ind]) % n
                    big_delta_paths.append((start_ind, end_ind))
            if old_R[(i + 1) % n] > old_R[i] - 1:
                start_ind = (i + 1) % n
                end_ind = (i + old_R[i] + 1) % n
                small_delta_paths.append((start_ind, end_ind))
        else:
            for j in range(old_R[(i + 1) % n], -1, -1):
                end_ind = (i + 1 + j) % n
                next_ind = (end_ind + 1) % n
                if old_L[end_ind] > j and old_L[end_ind] > old_L[next_ind] - 1:
                    start_ind = (end_ind - old_L[end_ind]) % n
                    big_delta_paths.append((start_ind, end_ind))
            if old_L[i] > old_L[(i + 1) % n] - 1:
                start_ind = (i - old_L[(i + 1) % n]) % n
                end_ind = i
                small_delta_paths.append((start_ind, end_ind))

        big_deltas_list.append(big_delta_paths)
        small_deltas_list.append(small_delta_paths)
        current_R = reduce_wheel(current_R, i)
        W_list.append(current_R)

    return W_list, indexes_reduced, big_deltas_list, small_deltas_list, directions_used


def generate_equations_strings(W_list, indexes_reduced, big_deltas_list, small_deltas_list) -> List[str]:
    """Human-readable rendering of the accumulated inequalities, in terms of
    deleted-edge lengths t_k and final regular-wheel weights h_v."""
    m = len(indexes_reduced)
    equations: List[str] = []

    def project(curr_x, curr_y, start_step):
        t_acc = []
        for s in range(start_step, m):
            n_s = len(W_list[s])
            i_s = indexes_reduced[s]
            next_x, next_y = pi(curr_x, i_s, n_s), pi(curr_y, i_s, n_s)
            if (next_y - next_x) % (n_s - 1) != (curr_y - curr_x) % n_s:
                t_acc.append(f"t_{s}")
            curr_x, curr_y = next_x, next_y
        return curr_x, curr_y, t_acc

    for k in range(m):
        if not big_deltas_list[k] and not small_deltas_list[k]:
            continue
        equations.append(f"--- Step k={k} (Reduced edge {indexes_reduced[k]}) ---")
        current_n = len(W_list[k])
        i_edge = indexes_reduced[k]
        for x, y in big_deltas_list[k]:
            fx, fy, t_acc = project(pi(x, i_edge, current_n), pi(y, i_edge, current_n), k + 1)
            t_side = f"t_{k}" + "".join(f" + {t}" for t in t_acc)
            equations.append(f"{t_side} < h_{fx} + h_{fy}")
        for x, y in small_deltas_list[k]:
            fx, fy, t_acc = project(pi(x, i_edge, current_n), pi(y, i_edge, current_n), k + 1)
            t_side = f"t_{k}" + "".join(f" + {t}" for t in t_acc)
            equations.append(f"h_{fx} + h_{fy} < {t_side}")

    return equations


def solve_system(
    W_list, indexes_reduced, big_deltas_list, small_deltas_list, *, silent: bool = True, iis: bool = False
) -> Dict[str, Any]:
    """Solve the t/h LP built from a completed reduction path.

    minimize sum(t) + sum(h)  s.t.  t_k, h_v >= 1, and for each recorded delta:
        big:   t_k <= (projected h-difference) - epsilon
        small: (projected h-difference) <= t_k - epsilon

    Set iis=True to compute and attach the Irreducible Inconsistent Subsystem when
    infeasible (replaces the separate solve_system_with_iis copy that only differed
    by naming its constraints so IIS reporting had something to point at).
    """
    m = len(indexes_reduced)
    final_R = W_list[-1]
    final_n = len(final_R)

    model = gp.Model("Wheel_Metrizability")
    if silent:
        model.setParam("OutputFlag", 0)

    t_vars = [model.addVar(vtype=GRB.CONTINUOUS, name=f"t_{k}", lb=1.0) for k in range(m)]
    h_vars = [model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{v}", lb=1.0) for v in range(final_n)]

    def delta_expr(curr_k: int, x: int, y: int):
        if curr_k == m:
            return h_vars[x] + h_vars[y]
        current_R = W_list[curr_k]
        i_s, n_s = indexes_reduced[curr_k], len(current_R)
        next_x, next_y = pi(x, i_s, n_s), pi(y, i_s, n_s)
        expr = delta_expr(curr_k + 1, next_x, next_y)
        if (next_y - next_x) % (n_s - 1) != (y - x) % n_s:
            expr = expr - t_vars[curr_k]
        return expr

    epsilon = 1e-4
    for k in range(m):
        current_n = len(W_list[k])
        i_edge = indexes_reduced[k]
        for x, y in big_deltas_list[k]:
            expr = delta_expr(k + 1, pi(x, i_edge, current_n), pi(y, i_edge, current_n))
            name = f"big_delta_k{k}_{x}_{y}" if iis else ""
            model.addConstr(t_vars[k] <= expr - epsilon, name=name)
        for x, y in small_deltas_list[k]:
            expr = delta_expr(k + 1, pi(x, i_edge, current_n), pi(y, i_edge, current_n))
            name = f"small_delta_k{k}_{x}_{y}" if iis else ""
            model.addConstr(expr <= t_vars[k] - epsilon, name=name)

    model.setObjective(gp.quicksum(t_vars) + gp.quicksum(h_vars), GRB.MINIMIZE)
    model.optimize()

    result: Dict[str, Any] = {
        "status": model.status,
        "is_metrizable": model.status == GRB.OPTIMAL,
        "t_vals": {},
        "h_vals": {},
        "iis_constraints": [],
    }
    if result["is_metrizable"]:
        result["t_vals"] = {k: round(t_vars[k].X, 4) for k in range(m)}
        result["h_vals"] = {v: round(h_vars[v].X, 4) for v in range(final_n)}
    elif iis and model.status == GRB.INFEASIBLE:
        result["iis_constraints"] = extract_iis(model)
    return result


def analyze_wheel(R_initial: List[int], *, silent_gurobi: bool = True) -> str:
    """Deterministic single-path analysis: reduce greedily rightward, report the
    accumulated inequalities and whether the resulting LP is feasible."""
    W_list, indexes_reduced, big_deltas, small_deltas = find_reduction_wheels(R_initial)

    report = [f"=== Analysis for Wheel R = {R_initial} ===", f"Performed {len(indexes_reduced)} reductions."]
    for k in range(len(indexes_reduced)):
        report.append(f"Step {k}: Reduced edge {indexes_reduced[k]}. New R = {W_list[k + 1]}")

    report.append("\n=== Final linear system (Translated to h_i) ===")
    equations = generate_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    report.extend(equations if equations else ["No constraints generated."])

    report.append("\n=== Gurobi Solution ===")
    sol = solve_system(W_list, indexes_reduced, big_deltas, small_deltas, silent=silent_gurobi)
    if sol["is_metrizable"]:
        report.append("Result: Metrizable System Found!")
        report.append("Deleted edge lengths (t_k):")
        report.extend(f"  t_{k} = {v}" for k, v in sol["t_vals"].items())
        report.append("\nFinal Regular Wheel variables (h_v):")
        report.extend(f"  h_{v} = {val}" for v, val in sol["h_vals"].items())
    else:
        report.append("Result: System is NOT Metrizable (Infeasible/Unbounded).")
    report.append("=" * 50 + "\n")
    return "\n".join(report)


def _dfs_all_reductions(R_initial: List[int], *, want_metrizable: bool, silent_gurobi: bool):
    """Shared DFS core for analyze_wheel_with_backtrack / hunt_infeasible_wheel:
    explores every combination of right/left reductions at every step.

    If want_metrizable: returns the first leaf whose LP is feasible (or None if
    every leaf is infeasible) plus the total number of leaves explored.
    If not want_metrizable: returns the first leaf's full (infeasible) result
    plus the count, or None if any leaf turns out feasible (i.e. the wheel is
    NOT provably infeasible).
    """
    paths_explored = [0]
    first_result = [None]

    def dfs(current_R, path):
        L = get_L_from_R(current_R)
        red_right = get_reducible_edges_right(current_R)
        red_left = get_reducible_edges_left(L)

        if not red_right and not red_left:
            paths_explored[0] += 1
            W_list, indexes_reduced, big_deltas, small_deltas, dirs = build_system_from_path(R_initial, path)
            sol = solve_system(W_list, indexes_reduced, big_deltas, small_deltas, silent=silent_gurobi, iis=not want_metrizable)
            leaf = (path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs)

            if want_metrizable:
                return leaf if sol["is_metrizable"] else None
            else:
                if sol["is_metrizable"]:
                    return True  # signal: this wheel is NOT provably infeasible
                if first_result[0] is None:
                    first_result[0] = leaf
                return False

        for i_r in red_right:
            res = dfs(reduce_wheel(current_R, i_r), path + [(i_r, "R")])
            if res:
                return res
        for i_l in red_left:
            res = dfs(reduce_wheel(current_R, i_l), path + [(i_l, "L")])
            if res:
                return res
        return None if want_metrizable else False

    outcome = dfs(R_initial, [])
    return outcome, paths_explored[0], first_result[0]


def analyze_wheel_with_backtrack(R_initial: List[int], *, silent_gurobi: bool = True) -> str:
    """Full combinatorial search over every left/right reduction order. Reports
    the first metrizable leaf found, or that the wheel is PROVEN NOT METRIZABLE if
    every leaf in the tree is infeasible."""
    result, paths_explored, _ = _dfs_all_reductions(R_initial, want_metrizable=True, silent_gurobi=silent_gurobi)

    report = [
        f"=== Combinatorial Backtracking Analysis for Wheel R = {R_initial} ===",
        f"Total paths explored in tree: {paths_explored}",
    ]
    if result is None:
        report.append("\nResult: System is PROVEN NOT METRIZABLE.")
        report.append("All possible combinations and orders of Left/Right reductions resulted in Infeasible models.")
        report.append("=" * 50 + "\n")
        return "\n".join(report)

    path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs = result
    path_str = " -> ".join(f"Edge {idx}({d})" for idx, d in path)
    report.append(f"\nSUCCESS! Metrizable sequence found:\n{path_str}")
    report.append(f"\nPerformed {len(indexes_reduced)} reductions.")
    for k in range(len(indexes_reduced)):
        report.append(f"Step {k} [{dirs[k]}]: Reduced edge {indexes_reduced[k]}. New R = {W_list[k + 1]}")

    report.append("\n=== Final linear system (Translated to h_i) ===")
    equations = generate_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    report.extend(equations if equations else ["No constraints generated."])

    report.append("\n=== Gurobi Solution ===")
    report.append("Deleted edge lengths (t_k):")
    report.extend(f"  t_{k} = {v}" for k, v in sol["t_vals"].items())
    report.append("\nFinal Regular Wheel variables (h_v):")
    report.extend(f"  h_{v} = {val}" for v, val in sol["h_vals"].items())
    report.append("=" * 50 + "\n")
    return "\n".join(report)


def hunt_infeasible_wheel(R_initial: List[int], *, silent_gurobi: bool = True) -> Optional[str]:
    """Explore the full reduction tree looking for a wheel that is infeasible on
    EVERY leaf. Returns None if any leaf is metrizable (i.e. the wheel is not
    provably infeasible); otherwise returns a report of the first-tried path and
    its IIS (the contradiction Gurobi found)."""
    is_metrizable, paths_explored, first_failure = _dfs_all_reductions(R_initial, want_metrizable=False, silent_gurobi=silent_gurobi)
    if is_metrizable or first_failure is None:
        return None

    path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs = first_failure
    path_str = " -> ".join(f"Edge {idx}({d})" for idx, d in path)
    report = [
        f"=== Infeasible-Wheel Hunt for R = {R_initial} ===",
        f"Total paths explored in tree: {paths_explored}",
        "Result: PROVEN INFEASIBLE on every reduction order.",
        f"\nFirst path tried:\n{path_str}",
        "\n=== IIS (contradiction) on that path ===",
    ]
    report.extend(f"  {c}" for c in sol["iis_constraints"])
    report.append("=" * 50 + "\n")
    return "\n".join(report)
