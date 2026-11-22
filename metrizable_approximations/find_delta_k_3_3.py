import networkx as nx
from typing import List, Tuple, Dict

from path_systems_lib.delta_search import find_delta
from path_systems_lib.graph_families import k_3_3
from path_systems_lib.path_weight_lp import check_alpha

# ==========================================
# מחולל מערכות שכניות קונסיסטנטיות
# ==========================================


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


def generate_neighborly_systems(G: nx.Graph) -> List[Dict[Tuple[int, int], List[int]]]:
    nodes = list(G.nodes())
    base_system = {}
    for u, v in G.edges():
        base_system[(u, v)] = [u, v]
        base_system[(v, u)] = [v, u]

    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                pairs.append((nodes[i], nodes[j]))

    all_simple_paths = {}
    for u, v in pairs:
        valid_paths = [p for p in nx.all_simple_paths(G, u, v) if is_consistent(p, base_system)]
        all_simple_paths[(u, v)] = valid_paths

    pairs.sort(key=lambda pair: len(all_simple_paths[pair]))
    valid_systems = []

    def backtrack(pair_idx: int, current_system: Dict[Tuple[int, int], List[int]]):
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

    print(f"Generating systems for graph with {G.number_of_nodes()} nodes...")
    backtrack(0, base_system)
    print(f"Found {len(valid_systems)} consistent neighborly systems.")
    return valid_systems


# ==========================================
# מציאת דלתא ואיתור הנעילה (via path_weight_lp / delta_search)
# ==========================================

def find_system_with_large_delta():
    # נתחיל מ-K_3,3
    G = k_3_3()
    systems = generate_neighborly_systems(G)

    for idx, sys in enumerate(systems):
        # בדיקה מהירה: האם היא מטריזבילית חזק? (דלתא 1.0)
        if check_alpha(G, sys, 1.0).feasible:
            continue

        # אם לא, נמצא את הדלתא המדויקת עם חיפוש בינארי
        result = find_delta(G, sys, hi=10.0, tol=1e-4, auto_expand_hi=False, compute_iis_at_result=False)
        delta = result.delta

        # אם מצאנו מערכת שבה הדלתא גדולה משמעותית מ-1 (למשל 1.1 ומעלה)
        if delta > 1.01:
            print("\n" + "=" * 50)
            print("🎯 BINGO! FOUND STRICTLY DOMINATED SYSTEM!")
            print(f"System #{idx + 1} | Delta(P) = {delta:.4f}")
            print("=" * 50)

            iis_result = check_alpha(G, sys, delta - 1e-4, compute_iis=True)
            print("\n🔒 The Interlocking Contradiction (IIS) at maximum failing alpha:")
            for constr in (iis_result.iis or []):
                print("  " + constr)

            print("\n📍 The Path System that caused it:")
            for (u, v), path in sys.items():
                if u < v and not G.has_edge(u, v):
                    print(f"  {u} -> {v}: {path}")
            return  # עוצרים אחרי שמצאנו את הראשונה

    print("All systems had Delta = 1. Trying a larger graph...")


if __name__ == "__main__":
    find_system_with_large_delta()
