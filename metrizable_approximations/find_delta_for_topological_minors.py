import networkx as nx
from typing import List, Tuple, Dict

from path_systems_lib.delta_search import find_delta
from path_systems_lib.path_weight_lp import build_alpha_model

# ==========================================
# Analysis (Family A LP, via path_weight_lp)
# ==========================================


def analyze_path_system(G: nx.Graph, defined_paths: Dict[Tuple[int, int], List[int]], tol: float = 1e-4) -> None:
    print("\n--- STEP 1: Checking Weak Metrizability ---")
    is_weak = build_alpha_model(G, defined_paths, alpha=1.0, eps=0.0).feasible

    if is_weak:
        print(">> System IS WEAKLY Metrizable. Checking Strong Metrizability...")
        strong_result = build_alpha_model(G, defined_paths, alpha=1.0, eps=1e-3, weight_ub=1000.0)

        if strong_result.feasible:
            print(">> System IS STRONGLY Metrizable!")
            print(">> Delta(P) = 1.0")
            for k, val in strong_result.weights.items():
                print(f"      w{k} = {val:.3f}")
        else:
            print(">> System is WEAKLY BUT NOT STRONGLY Metrizable.")
            print(">> Delta(P) = 1.0")
        return

    print("\n--- STEP 2: System is NOT Weakly Metrizable. Starting Binary Search for Delta(P) ---")

    result = find_delta(G, defined_paths, tol=tol, eps=0.0, auto_expand_hi=False)

    print(f"\n[+] Found minimal Delta(P) ≈ {result.delta:.5f}")
    if result.weights:
        print("    Valid weights for Delta(P):")
        for k, val in result.weights.items():
            print(f"      w{k} = {val:.3f}")

    failing_alpha = result.delta - tol
    print(f"\n[!] Extracting Contradiction (IIS) at failing alpha = {failing_alpha:.5f}...")
    iis_result = build_alpha_model(G, defined_paths, alpha=failing_alpha, eps=0.0, compute_iis=True)

    print("    Conflicting Constraints:")
    for constr in (iis_result.iis or []):
        print(f"      - {constr}")


# ==========================================
# Graph & Path System Definitions Library
# ==========================================

def get_graph_data(graph_id: int) -> Tuple[nx.Graph, Dict[Tuple[int, int], List[int]]]:
    """
    Returns the graph topology and the full consistent path system based on the ID.
    """
    G = nx.Graph()
    exceptional_paths = []

    if graph_id == 0:
        # Petersen Graph (Reference from previous analysis)
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        G.add_edges_from([(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)])
        G.add_edges_from([(6, 8), (8, 10), (10, 7), (7, 9), (9, 6)])

        exceptional_paths = [
            [1, 5, 10, 7],
            [5, 4, 9, 6],
            [4, 3, 8, 10],
            [3, 2, 7, 9],
            [2, 1, 6, 8]
        ]

    # --- FILL IN THE 16 MINORS HERE ---
    elif graph_id == 1:
        G.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 7), (6, 7)])
        exceptional_paths = [
            [1, 3, 2], [1, 6, 7], [2, 7, 6], [3, 1, 4], [3, 2, 5],
            [3, 1, 6], [3, 1, 6, 7], [4, 1, 5], [4, 2, 7, 6],
            [4, 2, 7], [5, 1, 6], [5, 2, 7]
        ]

    elif graph_id == 2:
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 6), (6, 3), (1, 7), (7, 4)])
        exceptional_paths = [
            [2, 3, 4, 5], [6, 3, 4, 7], [2, 1, 6], [5, 1, 7]
        ]

    elif graph_id == 3:
        G.add_edges_from([(1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 7)])
        exceptional_paths = [
            [1, 7, 4, 2], [5, 3, 4, 7], [5, 1, 6], [2, 6, 3]
        ]

    elif graph_id == 4:
        # Graph 4: Edges automatically inferred from the neighborly paths
        G.add_edges_from([
            (1, 2), (1, 5), (2, 3), (2, 6), (3, 4),
            (3, 6), (4, 5), (4, 7), (5, 7), (6, 7)
        ])
        exceptional_paths = [
            [1, 2, 3], [1, 2, 3, 4], [1, 2, 6], [1, 5, 7], [2, 3, 4],
            [2, 1, 5], [2, 1, 5, 7], [3, 4, 5], [3, 6, 7], [4, 7, 6],
            [5, 1, 2, 6]
        ]

    elif graph_id == 5:
        G.add_edges_from([(1, 2), (1, 5), (2, 3), (2, 6), (3, 4), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)])
        exceptional_paths = [
            [2, 1, 5, 4], [1, 5, 4, 3], [3, 2, 6], [2, 3, 7], [4, 7, 6], [5, 6, 7]
        ]

    elif graph_id == 6:
        exceptional_paths = [[1, 5, 4, 3], [1, 5, 4], [1, 6, 7], [1, 6, 8],
                              [2, 3, 4], [2, 3, 4, 5], [2, 1, 6], [2, 1, 6, 7], [2, 1, 6, 8], [3, 4, 5],
                              [3, 8, 6], [3, 4, 7], [4, 5, 1, 6], [4, 3, 8],
                              [5, 1, 6], [5, 4, 7], [5, 1, 6, 8], [7, 4, 3, 8]
                              ]

    elif graph_id == 7:
        exceptional_paths = [
            [1, 2, 3], [1, 7, 6, 5, 4], [1, 7, 6, 5], [1, 7, 6], [2, 3, 4],
            [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 1, 7], [2, 1, 8], [3, 4, 5],
            [3, 4, 5, 6], [3, 2, 1, 7], [4, 5, 6], [4, 5, 6, 7], [4, 3, 8],
            [5, 6, 7], [5, 4, 3, 8], [7, 6, 8]
        ]

    elif graph_id == 8:
        exceptional_paths = [
            [1, 2, 3], [1, 2, 3, 4], [1, 6, 5], [1, 6, 8, 7], [1, 6, 8],
            [2, 3, 4], [2, 1, 6, 5], [2, 1, 6], [2, 7, 8], [3, 4, 5],
            [3, 4, 5, 6], [3, 2, 7], [3, 2, 7, 8], [4, 5, 6], [4, 8, 7],
            [5, 4, 8, 7], [5, 4, 8]
        ]

    elif graph_id == 9:
        exceptional_paths = [
            [1, 6, 5, 4, 3], [1, 6, 5, 4], [1, 6, 5], [1, 7, 8], [2, 3, 4],
            [2, 3, 4, 5], [2, 1, 6], [2, 3, 4, 5, 8], [3, 4, 5], [3, 4, 5, 6],
            [3, 2, 7], [3, 4, 5, 8], [4, 5, 6], [4, 3, 2, 7], [4, 5, 8],
            [5, 8, 7], [6, 5, 8, 7], [6, 5, 8]
        ]

    elif graph_id == 10:
        exceptional_paths = [
            [1, 5, 4, 3], [1, 5, 4], [1, 5, 6], [2, 3, 4], [2, 6, 5], [2, 1, 7],
            [2, 1, 8], [3, 4, 5], [3, 2, 6], [3, 4, 5, 1, 7], [3, 4, 5, 1, 8],
            [4, 3, 2, 6], [4, 5, 1, 7], [4, 5, 1, 8], [5, 1, 7], [5, 1, 8], [7, 6, 8]
        ]

    elif graph_id == 11:
        exceptional_paths = [
            [1, 6, 2], [1, 6, 8, 7], [1, 6, 8], [2, 5, 7], [2, 5, 7, 8], [3, 1, 4],
            [3, 2, 5], [3, 2, 6], [3, 2, 5, 7], [3, 2, 5, 7, 8], [4, 2, 5],
            [4, 2, 6], [4, 2, 5, 7], [4, 2, 5, 7, 8], [5, 1, 6], [5, 7, 8], [6, 8, 7]
        ]

    elif graph_id == 12:
        exceptional_paths = [
            [1, 5, 2], [1, 3, 7], [1, 5, 8], [2, 4, 7], [2, 5, 8], [3, 2, 4],
            [3, 1, 5], [3, 1, 6], [3, 1, 5, 8], [4, 8, 5], [4, 1, 6], [5, 2, 6],
            [5, 1, 3, 7], [6, 1, 3, 7], [6, 2, 5, 8], [7, 3, 1, 5, 8]
        ]

    elif graph_id == 13:
        exceptional_paths = [
            [1, 8, 3], [1, 8, 3, 4], [1, 7, 6, 5], [1, 7, 6], [1, 7, 9], [2, 3, 4],
            [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 1, 7], [2, 1, 8], [2, 1, 7, 9],
            [3, 4, 5], [3, 4, 5, 6], [3, 8, 1, 7], [3, 4, 5, 9], [4, 5, 6],
            [4, 3, 8, 1, 7], [4, 3, 8], [4, 5, 9], [5, 6, 7], [5, 4, 3, 8],
            [6, 7, 1, 8], [6, 7, 9], [7, 1, 8], [8, 3, 4, 5, 9]
        ]

    elif graph_id == 14:
        exceptional_paths = [
            [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 7, 6], [1, 7, 6, 8],
            [1, 2, 9], [2, 3, 4], [2, 3, 4, 5], [2, 9, 7, 6], [2, 9, 7], [2, 3, 8],
            [3, 4, 5], [3, 8, 6], [3, 2, 9, 7], [3, 2, 9], [4, 5, 6], [4, 5, 6, 7],
            [4, 5, 6, 8], [4, 5, 6, 7, 9], [5, 6, 7], [5, 6, 8], [5, 6, 7, 9],
            [6, 7, 9], [7, 6, 8], [8, 3, 2, 9]
        ]

    elif graph_id == 15:
        exceptional_paths = [
            [1, 7, 8, 9, 2], [1, 3, 4], [1, 5, 6], [1, 7, 8], [1, 7, 8, 9],
            [2, 4, 3], [2, 6, 5], [2, 9, 8, 7], [2, 9, 8], [3, 1, 5], [3, 1, 5, 6],
            [3, 1, 7], [3, 1, 7, 8], [3, 4, 2, 9], [4, 2, 6, 5], [4, 2, 6],
            [4, 3, 1, 7], [4, 3, 1, 7, 8], [4, 2, 9], [5, 1, 7], [5, 1, 7, 8],
            [5, 1, 7, 8, 9], [6, 2, 9, 8, 7], [6, 2, 9, 8], [6, 2, 9], [7, 8, 9]
        ]

    elif graph_id == 16:
        exceptional_paths = [
            [1, 2, 3], [1, 2, 3, 4], [1, 6, 5], [1, 6, 7], [1, 2, 9], [1, 2, 9, 10],
            [2, 3, 4], [2, 1, 6, 5], [2, 1, 6], [2, 1, 6, 7], [2, 3, 8], [2, 9, 10],
            [3, 4, 5], [3, 4, 5, 6], [3, 4, 7], [3, 8, 10, 9], [3, 8, 10], [4, 5, 6],
            [4, 3, 8], [4, 3, 8, 10, 9], [4, 3, 8, 10], [5, 4, 7], [5, 6, 1, 8],
            [5, 6, 1, 2, 9], [5, 6, 1, 2, 9, 10], [6, 1, 8], [6, 1, 2, 9],
            [6, 1, 2, 9, 10], [7, 4, 3, 8], [7, 6, 1, 2, 9], [7, 6, 1, 2, 9, 10],
            [8, 10, 9]
        ]

    elif graph_id == 17:
        G = nx.circulant_graph(9, [1, 3])
        exceptional_paths = [[i, (i + 3) % 9, (i + 4) % 9, (i + 7) % 9] for i in range(9)]

    else:
        raise ValueError(f"Graph ID {graph_id} is not defined.")

    # --- AUTO EDGE EXTRACTION (Based on your neighborly logic) ---
    # We only auto-extract if edges weren't explicitly added (like in Petersen)
    if G.number_of_edges() == 0:
        edges_set = set()
        for path in exceptional_paths:
            for i in range(len(path) - 1):
                edges_set.add(tuple(sorted((path[i], path[i + 1]))))
        G.add_edges_from(list(edges_set))

    # Automatically derive all subpaths to ensure consistency
    defined_paths = {}
    for path in exceptional_paths:
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                subpath = path[i:j + 1]
                u, v = subpath[0], subpath[-1]

                if u > v:
                    u, v = v, u
                    subpath = subpath[::-1]

                defined_paths[(u, v)] = subpath

    return G, defined_paths


# ==========================================
# Main Execution Menu
# ==========================================

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter the graph ID to analyze (0 for Petersen, 1-16 for minors, 'q' to quit): ")

        if user_input.lower() == 'q':
            break

        try:
            graph_id = int(user_input)
            print("\n=====================================")
            print(f" ANALYZING GRAPH {graph_id}")
            print("=====================================")

            G, defined_paths = get_graph_data(graph_id)
            analyze_path_system(G, defined_paths)

        except ValueError as e:
            print(f"Invalid input or graph not defined yet. {e}")
