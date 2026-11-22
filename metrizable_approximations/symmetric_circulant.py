import time
from typing import Dict, List

import networkx as nx

from path_systems_lib.graph_families import circulant
from path_systems_lib.path_weight_lp import check_alpha as lib_check_alpha


def add_path_sym(n: int, path: List[int], sys_dict: Dict) -> Dict:
    """
    מנסה להוסיף מסלול ואת כל הסיבובים (Shifts) שלו למערכת.
    אם מתגלה סתירה (המסלול סותר משהו קיים, או אפילו סותר סיבוב של עצמו) - מחזיר None.
    אחרת מחזיר מילון חדש ומעודכן.
    """
    new_sys = sys_dict.copy()
    for k in range(n):
        shifted_path = [(x + k) % n for x in path]

        for i in range(len(shifted_path)):
            for j in range(i + 1, len(shifted_path)):
                u, v = shifted_path[i], shifted_path[j]
                sub = shifted_path[i:j + 1]

                if (u, v) in new_sys:
                    expected = new_sys[(u, v)]
                    if expected != sub and expected[::-1] != sub:
                        return None  # סתירה התגלתה!
                else:
                    new_sys[(u, v)] = sub
                    new_sys[(v, u)] = sub[::-1]
    return new_sys


def check_alpha(G: nx.Graph, system: Dict, alpha: float) -> bool:
    return lib_check_alpha(G, system, alpha, path_cutoff=5).feasible


def find_high_delta_symmetric_system(n: int, jumps: List[int]):
    print(f"\n[{time.strftime('%H:%M:%S')}] Exploring Graph C_{n}({jumps}) for high Deltas...")
    G = circulant(n, jumps)

    base_system = {}
    for u, v in G.edges():
        base_system[(u, v)] = [u, v]
        base_system[(v, u)] = [v, u]

    target_distances = [d for d in range(1, n // 2 + 1) if d not in jumps]

    paths_by_distance = []
    for d in target_distances:
        valid_paths = []
        # Cutoff של 4 או 5 מספיק לחלוטין כדי למצוא סירקולציה מתרחבת בגרפים הללו
        for p in nx.all_simple_paths(G, 0, d, cutoff=4):
            # סינון עצמי: האם המסלול הזה בכלל יכול להתקיים בעולם סימטרי מבלי לסתור את עצמו?
            if add_path_sym(n, p, base_system) is not None:
                valid_paths.append(p)

        valid_paths.sort(key=len, reverse=True)
        paths_by_distance.append(valid_paths)
        print(f"  Target {d}: Found {len(valid_paths)} self-consistent candidate paths.")

    # --- מנוע ה-Backtracking החדש והמהיר ---
    valid_systems_found = []

    def dfs(depth: int, current_sys: Dict):
        # אם כבר מצאנו מערכת עם דלתא גבוהה, נעצור כדי לחסוך זמן
        if len(valid_systems_found) > 0:
            return

        if depth == len(target_distances):
            # הגענו לקצה העץ, יש לנו מערכת מלאה וחוקית! נבדוק לה את הדלתא.
            if not check_alpha(G, current_sys, 1.0):
                low, high = 1.0, 10.0
                while high - low > 1e-3:
                    mid = (low + high) / 2.0
                    if check_alpha(G, current_sys, mid):
                        high = mid
                    else:
                        low = mid

                if high > 1.01:
                    valid_systems_found.append((current_sys, high))
            return

        # ננסה לשבץ מסלולים עבור היעד הנוכחי
        for path in paths_by_distance[depth]:
            next_sys = add_path_sym(n, path, current_sys)
            if next_sys is not None:  # אם חוקי, ממשיכים להעמיק (אם לא, הענף נגזם!)
                dfs(depth + 1, next_sys)

    # התחלת עץ החיפוש
    dfs(0, base_system)

    if valid_systems_found:
        best_sys, delta = valid_systems_found[0]
        print("=" * 50)
        print(f"🚀 SUCCESS! Found TRUE Delta on C_{n}({jumps})")
        print(f"   Delta(P) ≈ {delta:.3f}")
        print("=" * 50)
        print("Path Strategy (Rule for Node 0):")
        for d in target_distances:
            p = best_sys[(0, d)]
            print(f"  Target {d}: Path {p} (Length: {len(p) - 1})")
    else:
        print("No strictly dominated symmetric systems found.")


if __name__ == "__main__":
    find_high_delta_symmetric_system(13, [1, 3, 4])
    # find_high_delta_symmetric_system(17, [1, 4, 5]) # למקרה שתרצה להמשיך להגדיל!
