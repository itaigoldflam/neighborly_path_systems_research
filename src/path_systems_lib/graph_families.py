"""Graph constructors for every family studied across the experiment folders.

Each family here is genuinely distinct math, not duplicated logic — consolidating
them is about discoverability and eliminating repeated re-implementations of the
same construction (e.g. 4 independent copies of `build_theta_ab2` /
`build_standard_grid`), not about merging distinct algorithms.
"""
from typing import Iterable, List, Sequence, Tuple

import networkx as nx

Mask = Tuple[int, ...]


def wheel(n: int) -> nx.Graph:
    """The wheel graph W_n: a hub (vertex 0) connected to every vertex of an
    (n-1)-cycle (vertices 1..n-1). Thin wrapper around nx.wheel_graph for
    discoverability and to document the hub-is-vertex-0 convention used repo-wide."""
    return nx.wheel_graph(n)


def grid(rows: int, cols: int) -> nx.Graph:
    """A standard rows x cols 2D grid graph (no wraparound), vertices labeled (i, j)."""
    G = nx.Graph()
    G.add_nodes_from((i, j) for i in range(rows) for j in range(cols))
    for i in range(rows):
        for j in range(cols):
            if i + 1 < rows:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
    return G


def torus_grid(rows: int, cols: int) -> nx.Graph:
    """A rows x cols 2D grid graph with wraparound edges in both directions."""
    G = nx.Graph()
    G.add_nodes_from((i, j) for i in range(rows) for j in range(cols))
    for i in range(rows):
        for j in range(cols):
            G.add_edge((i, j), ((i + 1) % rows, j))
            G.add_edge((i, j), (i, (j + 1) % cols))
    return G


def theta_ab2(a: int, b: int) -> nx.Graph:
    """Theta(a, b, 2): an (a+b)-cycle (the 'rim') plus a center vertex joined to
    rim vertices 0 and a (the 'spokes'), so 0 and a are connected by three
    internally-disjoint paths of length a, b, and 2 (via the center)."""
    G = nx.Graph()
    n = a + b
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    center = n
    G.add_edge(0, center)
    G.add_edge(a, center)
    return G


def theta_general(a: int, b: int, c: int) -> nx.Graph:
    """Theta(a, b, c): two poles (vertices 0 and 1) joined by three internally
    disjoint paths of lengths a, b, and c."""
    G = nx.Graph()
    U, V = 0, 1
    G.add_nodes_from([U, V])
    next_node = 2

    def add_branch(length: int) -> None:
        nonlocal next_node
        if length == 1:
            G.add_edge(U, V)
            return
        prev = U
        for _ in range(length - 1):
            G.add_edge(prev, next_node)
            prev = next_node
            next_node += 1
        G.add_edge(prev, V)

    for length in (a, b, c):
        add_branch(length)
    return G


def circulant(n: int, jumps: Sequence[int]) -> nx.Graph:
    """The circulant graph on Z_n with connection set `jumps`. Thin wrapper around
    nx.circulant_graph for discoverability alongside the other family constructors."""
    return nx.circulant_graph(n, jumps)


def paley(p: int) -> nx.Graph:
    """The Paley graph of prime order p (p == 1 mod 4): vertices 0..p-1, with an
    edge between i and j iff i - j is a nonzero quadratic residue mod p."""
    if p % 4 != 1:
        raise ValueError("Paley graphs require p == 1 (mod 4).")
    squares = {(x * x) % p for x in range(1, p)}
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(i + 1, p):
            if (i - j) % p in squares:
                G.add_edge(i, j)
    return G


def k_3_3() -> nx.Graph:
    """The complete bipartite graph K_{3,3}, parts {0,1,2} and {3,4,5}."""
    G = nx.Graph()
    G.add_nodes_from(range(6))
    for i in range(3):
        for j in range(3, 6):
            G.add_edge(i, j)
    return G


def petersen() -> nx.Graph:
    """The Petersen graph. Thin wrapper around nx.petersen_graph for discoverability."""
    return nx.petersen_graph()


def barbell(m1: int, m2: int) -> nx.Graph:
    """Two cliques of size m1 joined by a path of m2 intermediate vertices. Thin
    wrapper around nx.barbell_graph."""
    return nx.barbell_graph(m1, m2)


def partial_wheel(n: int, spoke_mask: Iterable[int]) -> nx.Graph:
    """A partial wheel: an n-cycle on rim vertices 1..n (hub is vertex 0), with a
    hub-to-rim spoke edge (0, i+1) present only where spoke_mask[i] == 1.

    spoke_mask must have length n; index i controls the spoke to rim vertex i+1.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n + 1))
    for i in range(n):
        G.add_edge(i + 1, (i + 1) % n + 1)
    for i, has_spoke in enumerate(spoke_mask):
        if has_spoke:
            G.add_edge(0, i + 1)
    return G


def get_unique_spoke_masks(n: int) -> List[Mask]:
    """Every spoke mask of length n (with at least 2 spokes) up to rotation and
    reflection, one canonical representative per equivalence class, sorted by
    (number of spokes, canonical mask). Used to enumerate the distinct partial
    wheels of a given size without generating rotation/reflection duplicates."""
    seen_canonical = set()
    unique_masks = []
    for i in range(1, 2 ** n):
        mask = tuple((i >> j) & 1 for j in range(n))
        if sum(mask) < 2:
            continue
        rotations = [mask[k:] + mask[:k] for k in range(n)]
        reflections = [rot[::-1] for rot in rotations]
        canonical = min(rotations + reflections)
        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            unique_masks.append(canonical)
    unique_masks.sort(key=lambda m: (sum(m), m))
    return unique_masks


def mask_to_gap_sequence(mask: Mask) -> List[int]:
    """The cyclic sequence of gaps (in rim steps) between consecutive spokes of
    `mask`, e.g. spokes at positions 0 and 3 out of 5 give gaps [3, 2]."""
    n = len(mask)
    indices = [i for i, val in enumerate(mask) if val == 1]
    gaps = []
    for i in range(len(indices)):
        gap = (indices[(i + 1) % len(indices)] - indices[i]) % n
        if gap == 0 and len(indices) == 1:
            gap = n
        gaps.append(gap)
    return gaps


def mask_to_partition_string(mask: Mask) -> str:
    """Human-readable rendering of a spoke mask as its gap sequence, e.g. '[3, 2]'."""
    return str(mask_to_gap_sequence(mask))
    return G
