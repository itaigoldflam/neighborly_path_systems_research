import networkx as nx
from typing import List, Tuple, Dict

from path_systems_lib.delta_search import find_delta
from path_systems_lib.path_weight_lp import build_alpha_model

# ==========================================
# Graph & Path System Definitions Library
# ==========================================


def get_graph_data() -> Tuple[nx.Graph, Dict[Tuple[int, int], List[int]]]:
    G = nx.circulant_graph(9, [1, 3])

    # --- פונקציות עזר פנימיות למציאת המערכת השלמה ---
    def is_consistent_local(new_path: List[int], current_paths: Dict[Tuple[int, int], List[int]]) -> bool:
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

    nodes = list(G.nodes())
    base_system = {}
    # אתחול עם הצלעות (שכנות)
    for u, v in G.edges():
        base_system[(u, v)] = [u, v]
        base_system[(v, u)] = [v, u]

    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if not G.has_edge(nodes[i], nodes[j])]

    all_simple_paths = {}
    for u, v in pairs:
        # מגבילים אורך ל-4, וממיינים מהארוך לקצר כדי לאלץ דלתא גדולה מ-1
        valid_paths = [p for p in nx.all_simple_paths(G, u, v, cutoff=4) if is_consistent_local(p, base_system)]
        valid_paths.sort(key=len, reverse=True)
        all_simple_paths[(u, v)] = valid_paths

    pairs.sort(key=lambda pair: len(all_simple_paths[pair]))
    valid_systems = []

    # חיפוש בינארי שמוצא רק את המערכת החוקית הראשונה
    def backtrack(pair_idx: int, current_system: Dict[Tuple[int, int], List[int]]):
        if len(valid_systems) >= 1:
            return
        if pair_idx == len(pairs):
            valid_systems.append(current_system.copy())
            return

        u, v = pairs[pair_idx]
        for path in all_simple_paths[(u, v)]:
            if is_consistent_local(path, current_system):
                current_system[(u, v)] = path
                current_system[(v, u)] = path[::-1]
                backtrack(pair_idx + 1, current_system)
                del current_system[(u, v)]
                del current_system[(v, u)]

    backtrack(0, base_system)

    # סינון המערכת החוקית לפורמט (u,v) כאשר u < v כפי שהקוד שלך דורש
    full_system = valid_systems[0]
    defined_paths = {k: v for k, v in full_system.items() if k[0] < k[1]}

    return G, defined_paths


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
# Main Execution Menu
# ==========================================

if __name__ == "__main__":

    G, defined_paths = get_graph_data()
    analyze_path_system(G, defined_paths)
