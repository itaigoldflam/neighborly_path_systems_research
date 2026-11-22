import networkx as nx
from typing import List, Tuple, Dict
from tqdm import tqdm

from path_systems_lib.delta_search import find_delta
from path_systems_lib.graph_families import barbell, circulant
from path_systems_lib.path_weight_lp import check_alpha

# ==========================================
# 1. פונקציות ליצירת המשפחות הלא-סימטריות
# ==========================================


def get_circulant_graphs() -> List[Tuple[str, nx.Graph]]:
    """
    מייצר רשימה של גרפים צירקולנטיים קטנים, שהם מועמדים מעולים
    למערכות שאינן מישוריות ויש בהן פוטנציאל לסירקולציה מתרחבת.
    """
    graphs = []
    # נתחיל מגדולים יחסית כדי שיהיה קוטר גדול מ-2, אבל קטנים מספיק להרצה מהירה
    graphs.append(("Circulant(n=8, jumps=[1,3])", circulant(8, [1, 3])))
    graphs.append(("Circulant(n=9, jumps=[1,4])", circulant(9, [1, 4])))
    graphs.append(("Circulant(n=10, jumps=[1,3,4])", circulant(10, [1, 3, 4])))
    return graphs


def get_barbell_graphs() -> List[Tuple[str, nx.Graph]]:
    """
    מייצר גרפי 'משקולת'. שני קליקים שמחוברים ביניהם בצוואר בקבוק,
    מה שמכריח מסלולים מסוימים להיות ארוכים מאוד ביחס לחלופות.
    """
    graphs = []
    # Barbell של שני משולשים המחוברים בצלע יחידה
    graphs.append(("Barbell(3,0) - Two Triangles", barbell(3, 0)))
    # Barbell של שני משולשים המחוברים במסלול של צומת אחד (קוטר גדול יותר)
    graphs.append(("Barbell(3,1) - Triangles with bridge", barbell(3, 1)))
    # Barbell של שני K4 מחוברים בצלע
    graphs.append(("Barbell(4,0) - Two K4s", barbell(4, 0)))
    return graphs


# ==========================================
# 2. מנוע יצירת מערכות מסלולים שכניות חוקיות
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


def generate_first_k_neighborly_systems(G: nx.Graph, max_systems: int = 500) -> List[Dict[Tuple[int, int], List[int]]]:
    """
    מייצר מערכות שכניות חוקיות. כדי למנוע זמן ריצה אינסופי בגרפים סבוכים,
    נעצור אחרי שמצאנו כמות מסוימת של מערכות.
    """
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
        valid_paths = [p for p in nx.all_simple_paths(G, u, v, cutoff=4) if is_consistent(p, base_system)]
        all_simple_paths[(u, v)] = valid_paths

    # Heuristic: start with pairs that have the fewest path choices
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


# ==========================================
# 3. מנוע Gurobi למציאת הדלתא (via path_weight_lp / delta_search)
# ==========================================

# ==========================================
# 4. התוכנית הראשית - סורק המשפחות
# ==========================================

def run_delta_hunter():
    print("Starting the hunt for an Expanding Circulation (Delta > 1)...")

    graphs_to_test = get_barbell_graphs()

    for graph_name, G in graphs_to_test:
        print(f"\nAnalyzing Graph: {graph_name} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")

        # מחפשים עד 500 מערכות בכל גרף
        systems = generate_first_k_neighborly_systems(G, max_systems=500)
        print(f"Generated {len(systems)} valid neighborly path systems. Now checking for high-delta systems...")

        for idx, sys in enumerate(tqdm(systems, desc=f"Scanning {graph_name}", unit="sys")):
            # בדיקת סינון מהירה: האם פתיר ב-1.0?
            if check_alpha(G, sys, 1.0).feasible:
                continue

            # אם לא פתיר, נמצא את הדלתא עם חיפוש בינארי
            result = find_delta(G, sys, hi=10.0, tol=1e-4)
            delta = result.delta

            # האם מצאנו את גביע הקודש?
            if delta > 1.01:
                print("\n" + "★" * 60)
                print(f"BINGO! Found a Strictly Dominated System in {graph_name}!")
                print(f"System #{idx + 1} | Delta(P) = {delta:.4f}")
                print("★" * 60)

                iis_result = check_alpha(G, sys, delta - 1e-4, compute_iis=True)
                iis = iis_result.iis or []

                print("\n[ IIS CONSTRAINTS ] - The locked contradiction:")
                for constr in iis:
                    print("  " + constr)

                print("\n[ PREFERRED PATHS IN IIS ]:")
                # נדפיס רק את המסלולים שמשתתפים בסתירה כדי לא להעמיס
                for (u, v), path in sys.items():
                    if u < v and not G.has_edge(u, v):
                        # נבדוק אם הצמתים האלה מופיעים באחד ממשפטי ה-IIS
                        path_str = f"path_{u}_{v}"
                        path_str_rev = f"path_{v}_{u}"
                        if any(path_str in c or path_str_rev in c for c in iis):
                            print(f"  {u} -> {v}: {path}")

                return  # עוצרים אחרי מציאת הדוגמה הראשונה כדי שתוכל לנתח אותה

        print("-> All tested systems in this graph had Delta = 1.0.")


if __name__ == "__main__":
    run_delta_hunter()
