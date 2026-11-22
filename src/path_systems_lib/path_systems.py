"""Enumeration of neighborly path systems of a graph.

Two genuinely different algorithms for the same underlying question ("what are
the neighborly path systems of G?") existed side by side in the repo and are both
kept here as named strategies, since they have different performance profiles and
one (the clique-based version) does not support a length cutoff at all:

- `generate_neighborly_systems`: a parameterized backtracking search, consolidating
  ~15 near-identical copies scattered across metrizable_grid_on_torus/, partial_wheels/,
  theta_graphs/, and metrizable_approximations/. It materializes every subpath of a
  candidate path into the working system as soon as it's chosen (the same idea
  path_tree_systems.PathSystem uses internally), which eliminates by construction a
  class of "reversed-path" consistency bugs that existed in 3 of the 4 original grid
  files (only one of which had patched it locally with an explicit `!= path[::-1]`
  check). Supports a path-length cutoff, multiple pair/candidate orderings, and a cap
  on how many systems to collect.
- `get_neighborly_path_systems_exact`: an exact enumerator via tree-consistency-graph
  clique-finding (metrizable_wheels/utils/FindingPathSystems.py's approach, moved here
  near-verbatim since it was already the most general, correct implementation). No
  cutoff support, but finds every *maximal* neighborly system exactly; used for wheels.
"""
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .path_tree_systems import (
    PathSystem,
    RootedTree,
    getSubpaths,
    orderPath,
)

Pair = Tuple[int, int]
Path = Tuple[int, ...]


def generate_neighborly_systems(
    G: nx.Graph,
    *,
    cutoff: Optional[int] = None,
    length_tolerance: Optional[int] = None,
    max_systems: Optional[int] = 1,
    order: str = "shortest_first",
    candidate_order: str = "natural",
    seed_with_edges: bool = True,
    stop_at_first: bool = False,
) -> List[Dict[Pair, Path]]:
    """Backtracking enumerator of neighborly path systems of G.

    Parameters
    ----------
    cutoff: max path length handed to nx.all_simple_paths (None = unbounded).
    length_tolerance: if set, overrides `cutoff` with a *per-pair* cutoff of
        shortest_path_length(u, v) + length_tolerance, matching the original
        metrizable_grid_on_torus scripts' `length_tolerance` semantics exactly
        (length_tolerance=0 there means "only consider shortest paths").
    max_systems: stop once this many systems have been found (None = exhaustive).
    order: how to order the vertex pairs the backtracker decides, one of
        "shortest_first" (ascending graph distance; matches the original
        metrizable_grid_on_torus/partial_wheels behavior),
        "mrv" (fewest candidate paths first; matches the original theta_graphs/
        metrizable_approximations behavior).
    candidate_order: how to order each pair's candidate paths, one of
        "natural" (whatever nx.all_simple_paths yields), "longest_first" (used by
        several metrizable_approximations scripts specifically to bias the search
        toward delta > 1 witnesses).
    seed_with_edges: if True (default), every graph edge is pre-committed to its
        trivial 2-vertex path before the search starts and excluded from the
        backtracked-over pairs (the "neighborly" constraint). If False, adjacent
        vertex pairs are also decided by the search like any other pair, matching
        the one file in the repo (find_deltas_for_partial_wheels.py) that
        deliberately leaves edges unconstrained.
    stop_at_first: shortcut for max_systems=1.

    Returns a list of plain dicts, one per found system, mapping each ordered pair
    (u, v) with u < v to the oriented path connecting them (path[0] == u). This
    includes not just the "top level" pairs but every subpath pair implied along
    the way (matching how every original copy's `current_paths`/`chosen_paths` dict
    was actually consumed downstream).
    """
    if order not in ("shortest_first", "mrv"):
        raise ValueError(f"Unknown order: {order!r}")
    if candidate_order not in ("natural", "longest_first"):
        raise ValueError(f"Unknown candidate_order: {candidate_order!r}")
    if stop_at_first:
        max_systems = 1

    nodes = list(G.nodes())
    mapping: Dict[Pair, Path] = {}
    if seed_with_edges:
        for u, v in G.edges():
            pair = orderPath((u, v))
            mapping[pair] = pair

    all_pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
    pending_pairs = [p for p in all_pairs if p not in mapping] if seed_with_edges else all_pairs

    candidates: Dict[Pair, List[List[int]]] = {}
    for u, v in pending_pairs:
        pair_cutoff = cutoff
        if length_tolerance is not None:
            pair_cutoff = nx.shortest_path_length(G, u, v) + length_tolerance
        paths = list(nx.all_simple_paths(G, u, v, cutoff=pair_cutoff))
        if candidate_order == "longest_first":
            paths.sort(key=len, reverse=True)
        candidates[(u, v)] = paths

    if order == "shortest_first":
        pending_pairs.sort(key=lambda p: nx.shortest_path_length(G, p[0], p[1]))
    else:
        pending_pairs.sort(key=lambda p: len(candidates[p]))

    results: List[Dict[Pair, Path]] = []

    def try_add(path: List[int]) -> Optional[Dict[Pair, Path]]:
        """Return the new (pair -> subpath) entries `path` would contribute, or
        None if any of its subpaths conflicts with an entry already in `mapping`."""
        new_entries: Dict[Pair, Path] = {}
        for pair, seg in getSubpaths(path).items():
            existing = mapping.get(pair)
            if existing is not None:
                if existing != seg:
                    return None
            else:
                new_entries[pair] = seg
        return new_entries

    def backtrack(idx: int) -> None:
        if max_systems is not None and len(results) >= max_systems:
            return
        if idx == len(pending_pairs):
            results.append(dict(mapping))
            return
        pair = pending_pairs[idx]
        if pair in mapping:
            # Already pinned down as a subpath of an earlier (longer) chosen path.
            backtrack(idx + 1)
            return
        for path in candidates[pair]:
            new_entries = try_add(path)
            if new_entries is None:
                continue
            mapping.update(new_entries)
            backtrack(idx + 1)
            for key in new_entries:
                del mapping[key]
            if max_systems is not None and len(results) >= max_systems:
                return

    backtrack(0)
    return results


# ---------------------------------------------------------------------------
# Exact enumeration via tree-consistency-graph clique-finding.
# Moved near-verbatim from metrizable_wheels/utils/FindingPathSystems.py, which
# was already the cleanest, most general implementation of this approach.
# ---------------------------------------------------------------------------

def _is_connected(G: nx.Graph) -> bool:
    v = next(iter(G.nodes))
    stack = [v]
    discovered = set()
    while stack:
        u = stack.pop()
        if u not in discovered:
            discovered.add(u)
            stack.extend(G.neighbors(u))
    return len(discovered) == len(G.nodes)


def _find_consistent_spanning_trees(G: nx.Graph, base_tree: RootedTree, constraints: PathSystem) -> List[RootedTree]:
    """All spanning trees of G containing base_tree whose induced path system does
    not conflict with `constraints`."""
    root = base_tree.root
    if len(base_tree.nodes) == len(G.nodes):
        return [base_tree]

    to_expand = [(base_tree, frozenset())]
    completed = []

    while to_expand:
        subtree, forbidden = to_expand.pop()
        viable_edges = set()
        for v in subtree.nodes:
            for u in set(G.neighbors(v)) - subtree.nodes:
                viable_edges.add(tuple(sorted((u, v))))
        viable_edges -= forbidden

        for edge in viable_edges:
            v1, v2 = edge
            v, v_new = (v1, v2) if v1 in subtree.nodes else (v2, v1)
            root_path = [v]
            u = v
            while u != root:
                u = subtree.predecessor(u)
                root_path.append(u)

            feasible = True
            for i, u in enumerate(root_path):
                pair = tuple(sorted((v_new, u)))
                if pair in constraints:
                    new_path = tuple(orderPath([v_new] + root_path[:i + 1]))
                    if constraints[pair] != new_path:
                        feasible = False
                        break

            if feasible:
                new_subtree = deepcopy(subtree)
                new_subtree.addNode(v, v_new)
                if len(new_subtree.nodes) == len(G.nodes):
                    completed.append(new_subtree)
                else:
                    to_expand.append((new_subtree, forbidden))

            forbidden = frozenset(forbidden | {edge})
            H = nx.Graph()
            H.add_nodes_from(G.nodes)
            H.add_edges_from(set(G.edges) - forbidden)
            if not _is_connected(H):
                break

    return completed


def _induced_path_system(tree: RootedTree) -> PathSystem:
    root = tree.root
    system = PathSystem(consistent=False)
    for v in tree.nodes - {root}:
        path = []
        u = v
        while u != root:
            u = tree.predecessor(u)
            path.append(u)
        for i, u in enumerate(path):
            system[orderPath((u, v))] = orderPath(tuple([v] + path[:i + 1]))
    return system


def _pairwise_consistent(system1: PathSystem, system2: PathSystem) -> bool:
    for pair in set(system1.keys()) & set(system2.keys()):
        if system1[pair] != system2[pair]:
            return False
    return True


def _tree_consistency_graph(G: nx.Graph):
    constraints = PathSystem(consistent=False)
    for edge in G.edges:
        constraints[orderPath(edge)] = orderPath(edge)

    rooted_trees = {}
    for v in G.nodes:
        base_tree = RootedTree(v)
        for u in G.neighbors(v):
            base_tree.addNode(v, u)
        rooted_trees[v] = _find_consistent_spanning_trees(G, base_tree, constraints)

    tree_systems = {v: [_induced_path_system(t) for t in trees] for v, trees in rooted_trees.items()}

    consistency_graph = {
        v: {i: {u: set() for u in set(G.nodes) - {v}} for i in range(len(trees))}
        for v, trees in rooted_trees.items()
    }
    for u, v in combinations(G.nodes, 2):
        for i, v_system in enumerate(tree_systems[v]):
            for j, u_system in enumerate(tree_systems[u]):
                if _pairwise_consistent(v_system, u_system):
                    consistency_graph[v][i][u].add(j)
                    consistency_graph[u][j][v].add(i)

    return consistency_graph, tree_systems


def _find_cliques(G: nx.Graph, consistency_graph) -> List[Dict]:
    vtxs = sorted(G.nodes, reverse=True, key=lambda v: G.degree[v])
    n = len(vtxs)
    if n == 1:
        return [{vtxs[0]: i} for i in consistency_graph[vtxs[0]].keys()]

    to_expand = [{vtxs[0]: i} for i in consistency_graph[vtxs[0]].keys()]
    max_cliques = []

    while to_expand:
        clq = to_expand.pop()
        k = len(clq)
        common = consistency_graph[vtxs[0]][clq[vtxs[0]]][vtxs[k]]
        for v in clq:
            common = common & consistency_graph[v][clq[v]][vtxs[k]]
        for i in common:
            new_clq = dict(clq)
            new_clq[vtxs[k]] = i
            if k + 1 == n:
                max_cliques.append(new_clq)
            else:
                to_expand.append(new_clq)

    return max_cliques


def get_neighborly_path_systems_exact(G: nx.Graph) -> List[PathSystem]:
    """Every maximal neighborly path system of G, found exactly via tree-consistency-
    graph clique enumeration (no length cutoff, no cap on count — use
    `generate_neighborly_systems` instead if G is large enough that this is too slow)."""
    consistency_graph, tree_systems = _tree_consistency_graph(G)
    max_cliques = _find_cliques(G, consistency_graph)

    all_systems = []
    for clique in max_cliques:
        system = PathSystem(consistent=False)
        defined: set = set()
        for v, tree_index in clique.items():
            v_system = tree_systems[v][tree_index]
            for pair in set(v_system.keys()) - defined:
                system[pair] = v_system[pair]
                defined.add(pair)
        system.consistent = True
        all_systems.append(system)

    return all_systems
