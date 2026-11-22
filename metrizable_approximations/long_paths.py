import random
from typing import List, Tuple, Dict

import networkx as nx

from path_systems_lib.delta_search import find_delta
from path_systems_lib.graph_families import circulant, petersen
from path_systems_lib.path_weight_lp import check_alpha


def get_test_graphs() -> List[Tuple[str, nx.Graph]]:
    return [
        ("Petersen Graph", petersen()),
        ("Circulant(8, [1,2])", circulant(8, [1, 2])),
        ("Circulant(9, [1,3])", circulant(9, [1, 3])),
    ]


def is_consistent(new_path: List[int], current_paths: Dict[Tuple[int, int], List[int]]) -> bool:
    length = len(new_path)
    for i in range(length):
        for j in range(i + 1, length):
            sub_u, sub_v = new_path[i], new_path[j]
            if (sub_u, sub_v) in current_paths:
                expected_subpath = current_paths[(sub_u, sub_v)]
                actual_subpath = new_path[i:j + 1]
                if actual_subpath != expected_subpath and actual_subpath != expected_subpath[::-1]:
                    return False
    return True


def generate_weird_neighborly_systems(G: nx.Graph, max_systems: int = 50) -> List[Dict[Tuple[int, int], List[int]]]:
    nodes = list(G.nodes())
    base_system = {}
    for u, v in G.edges():
        base_system[(u, v)] = [u, v]
        base_system[(v, u)] = [v, u]

    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if not G.has_edge(nodes[i], nodes[j])]

    all_simple_paths = {}
    for u, v in pairs:
        # מגבילים אורך כדי לא לקרוס, אבל מרשים מסלולים ארוכים מספיק כדי לייצר דלתא
        valid_paths = [p for p in nx.all_simple_paths(G, u, v, cutoff=4) if is_consistent(p, base_system)]

        # 💥 התיקון הקריטי: ממיינים מהארוך לקצר!
        # זה מכריח את המערכת להעדיף מסלולים ארוכים ולייצר את הסתירה שמובילה לדלתא > 1
        valid_paths.sort(key=len, reverse=True)

        all_simple_paths[(u, v)] = valid_paths

    pairs.sort(key=lambda pair: len(all_simple_paths[pair]))
    valid_systems = []

    def backtrack(pair_idx: int, current_system: Dict[Tuple[int, int], List[int]]):
        if len(valid_systems) >= max_systems:
            return
        if pair_idx == len(pairs):
            valid_systems.append(current_system.copy())
            return

        u, v = pairs[pair_idx]
        for path in all_simple_paths[(u, v)]:
            if is_consistent(path, current_system):
                current_system[(u, v)] = path
                current_system[(v, u)] = path[::-1]
                backtrack(pair_idx + 1, current_system)
                del current_system[(u, v)]
                del current_system[(v, u)]

    backtrack(0, base_system)
    return valid_systems


def run_fast_delta_hunter():
    print("Hunting for Strictly Dominated Systems (Long-Path First Heuristic)...")
    for graph_name, G in get_test_graphs():
        print(f"\nAnalyzing: {graph_name}")
        systems = generate_weird_neighborly_systems(G, max_systems=50)  # מספיק לבדוק מעט כי הן ממוינות!

        for idx, sys in enumerate(systems):
            if check_alpha(G, sys, 1.0, path_cutoff=4).feasible:
                continue

            result = find_delta(G, sys, hi=5.0, tol=1e-4, path_cutoff=4)
            delta = result.delta

            if delta > 1.01:
                print("\n" + "🚀" * 15)
                print(f"BINGO! Found Delta > 1 in {graph_name}!")
                print(f"System #{idx + 1} | Delta(P) = {delta:.4f}")
                print("🚀" * 15)

                iis_result = check_alpha(G, sys, delta - 1e-4, path_cutoff=4, compute_iis=True)
                print("\n[ IIS CONSTRAINTS (The lock) ]:")
                for constr in (iis_result.iis or []):
                    print("  " + constr)
                return

        print("-> All tested systems had Delta = 1.0")


if __name__ == "__main__":
    run_fast_delta_hunter()
