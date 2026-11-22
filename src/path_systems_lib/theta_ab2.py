"""Theta(a, b, 2)-specific path-system enumeration and parameter extraction.

Consolidates logic that was duplicated identically across 3-4 files in
theta_graphs/ (theta_ab2_graph_LP_solve.py, theta_ab2_check_metric.py,
theta_ab2_check_RL.py, generate_all_RL_vectors_theta_ab2.py): a backtracking
enumerator that hardcodes the theta(a,b,2)-specific structural constraint (the
0-a pair must route through the center), and extraction of the (R, L, k_A, k_B)
combinatorial parameters that characterize a solved system. Kept separate from
the generic path_systems.py backtracker since the center-routing rule is
domain-specific to this one graph family, not a general parameter.
"""
from typing import Dict, List, Tuple

import networkx as nx

from .graph_families import theta_ab2

Pair = Tuple[int, int]
Path = List[int]
System = Dict[Pair, Path]


def get_consistent_systems(a: int, b: int) -> Tuple[List[System], nx.Graph]:
    """Every consistent path system of Theta(a, b, 2), where the 0-a pair is
    forced to route through the center (the graph's defining structural property)."""
    G = theta_ab2(a, b)
    N = a + b
    center = N
    pairs = []
    paths_dict: Dict[Pair, List[Path]] = {}

    for u_node in range(N + 1):
        for v_node in range(u_node + 1, N + 1):
            pair = (u_node, v_node)
            pairs.append(pair)
            if G.has_edge(u_node, v_node):
                paths_dict[pair] = [[u_node, v_node]]
            elif pair == (0, a):
                paths_dict[pair] = [[0, center, a]]
            else:
                paths_dict[pair] = list(nx.all_simple_paths(G, u_node, v_node))

    pairs.sort(key=lambda p: len(paths_dict[p]))
    valid_systems: List[System] = []

    def can_add(system: System, new_path: Path) -> bool:
        for i in range(len(new_path)):
            for j in range(i + 1, len(new_path)):
                sub_pair = tuple(sorted((new_path[i], new_path[j])))
                if sub_pair in system:
                    chosen = system[sub_pair]
                    expected = new_path[i:j + 1]
                    if chosen != expected and chosen != expected[::-1]:
                        return False

        new_endpoints = (new_path[0], new_path[-1])
        for existing_path in system.values():
            if new_endpoints[0] in existing_path and new_endpoints[1] in existing_path:
                idx1 = existing_path.index(new_endpoints[0])
                idx2 = existing_path.index(new_endpoints[1])
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                expected = existing_path[idx1:idx2 + 1]
                if expected != new_path and expected != new_path[::-1]:
                    return False
        return True

    def backtrack(pair_idx: int, current_system: System) -> None:
        if pair_idx == len(pairs):
            valid_systems.append(current_system.copy())
            return
        pair = pairs[pair_idx]
        for path in paths_dict[pair]:
            if can_add(current_system, path):
                current_system[pair] = path
                backtrack(pair_idx + 1, current_system)
                del current_system[pair]

    backtrack(0, {})
    return valid_systems, G


def extract_parameters(system: System, a: int, b: int) -> Tuple[List[int], List[int], int, int]:
    """Extract the (R, L, k_A, k_B) combinatorial parameters that characterize
    a solved Theta(a, b, 2) path system: R/L are the clockwise/counter-clockwise
    rim reach vectors, k_A/k_B mark where the spanning tree of the center switches
    between routing via the two spokes."""
    N = a + b
    center = N
    R = [0] * N
    L = [0] * N

    for i in range(N):
        for step in range(1, N):
            target = (i + step) % N
            expected = [(i + k) % N for k in range(step + 1)]
            pair = tuple(sorted((i, target)))
            chosen = system[pair]
            if chosen == expected or chosen == expected[::-1]:
                R[i] = step
            else:
                break
        for step in range(1, N):
            target = (i - step) % N
            expected = [(i - k) % N for k in range(step + 1)]
            pair = tuple(sorted((i, target)))
            chosen = system[pair]
            if chosen == expected or chosen == expected[::-1]:
                L[i] = step
            else:
                break

    k_A = 0
    for i in range(a + 1):
        pair = tuple(sorted((i, center)))
        expected = [(i - k) for k in range(i + 1)] + [center]
        if system[pair] == expected or system[pair] == expected[::-1]:
            k_A = i

    k_B = a
    for i in range(a, N + 1):
        node = i % N
        pair = tuple(sorted((node, center)))
        expected = [(i - k) % N for k in range(i - a + 1)] + [center]
        if system[pair] == expected or system[pair] == expected[::-1]:
            k_B = i

    return R, L, k_A, k_B
