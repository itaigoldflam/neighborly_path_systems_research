import networkx as nx
from typing import List, Tuple, Dict
from tqdm import tqdm

from path_systems_lib.delta_search import find_delta
from path_systems_lib.graph_families import get_unique_spoke_masks, mask_to_gap_sequence, partial_wheel
from path_systems_lib.io_utils import output_dir
from path_systems_lib.path_weight_lp import check_alpha


def is_interesting_partition(mask: Tuple[int, ...]) -> bool:
    """
    מסנן חלוקות לפי התנאים שביקשת:
    1. בדיוק 3 מספרים >= 2, ואחד מהם >= 3.
    2. בדיוק 4 מספרים >= 2.
    """
    distances = mask_to_gap_sequence(mask)
    # מוצאים את כל המקטעים שהם בגודל 2 ומעלה
    large_segments = [x for x in distances if x >= 2]
    num_large = len(large_segments)

    if num_large == 4:
        return True

    if num_large == 3 and any(x >= 3 for x in large_segments):
        return True

    return False


def mask_to_partition_string(mask: Tuple[int, ...]) -> str:
    return "[" + ",".join(map(str, mask_to_gap_sequence(mask))) + "]"


# ==========================================
# ג'ינרוט מערכות מסלולים קונסיסטנטיות
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


def generate_all_path_systems(G: nx.Graph) -> List[Dict[Tuple[int, int], List[int]]]:
    nodes = list(G.nodes())

    # 1. אכיפת תנאי השכניות: אתחול המערכת עם כל הצלעות הישירות בגרף
    base_system = {}
    for u, v in G.edges():
        base_system[(u, v)] = [u, v]
        base_system[(v, u)] = [v, u]

    # 2. איסוף הזוגות שעבורם צריך לבחור מסלול (רק קודקודים ש*אינם* שכנים)
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                pairs.append((nodes[i], nodes[j]))

    # 3. מציאת מסלולים פשוטים לכל זוג וסינון ראשוני מחמיר
    all_simple_paths = {}
    for u, v in pairs:
        # שומרים רק מסלולים שלא סותרים את בחירת הצלעות הישירות של הגרף!
        valid_paths_for_pair = [
            p for p in nx.all_simple_paths(G, u, v)
            if is_consistent(p, base_system)
        ]
        all_simple_paths[(u, v)] = valid_paths_for_pair

    # 4. אופטימיזציה קריטית ל-Backtracking:
    # מתחילים מהזוגות שיש להם הכי מעט אופציות למסלולים (Minimum Remaining Values)
    pairs.sort(key=lambda pair: len(all_simple_paths[pair]))

    valid_systems = []

    pbar = tqdm(desc="Generating Neighborly Systems", position=0, leave=True, unit="sys")

    def backtrack(pair_idx: int, current_system: Dict[Tuple[int, int], List[int]]):
        if pair_idx == len(pairs):
            valid_systems.append(current_system.copy())
            pbar.update(1)
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
    pbar.close()
    return valid_systems


# ==========================================
# בדיקת דלתא ומציאת IIS (via path_weight_lp / delta_search)
# ==========================================

def get_system_delta(G: nx.Graph, system: Dict[Tuple[int, int], List[int]]) -> Tuple[float, List[str]]:
    if check_alpha(G, system, 1.0).feasible:
        return 1.0, []

    result = find_delta(G, system, hi=10.0, tol=1e-4, max_hi=1000, compute_iis_at_result=True)
    return result.delta, (result.iis or [])


# ==========================================
# המנוע הראשי
# ==========================================

def main():
    n = int(input("Enter the size of the partial wheel (n): "))
    output_path = output_dir(__file__) / f"neighborly_partial_wheels_delta_n_{n}.txt"

    all_partitions = get_unique_spoke_masks(n)

    # --- הוספת הסינון כאן ---
    partitions = [p for p in all_partitions if is_interesting_partition(p)]

    print(f"\nFound {len(all_partitions)} unique partitions for n={n}.")
    print(f"Filtered down to {len(partitions)} interesting partitions to analyze.\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"DELTA RESEARCH RESULTS FOR PARTIAL WHEELS (n={n})\n")
        f.write("Filtered to specific configurations (3 larges with one >=3, or 4 larges)\n")
        f.write("=" * 60 + "\n\n")

        for part in partitions:
            part_str = mask_to_partition_string(part)
            print(f"\n---> Analyzing Partition: {part_str} <---")
            G = partial_wheel(n, part)

            systems = generate_all_path_systems(G)

            if not systems:
                print("No consistent path systems found.")
                f.write(f"Partition {part_str}:\nSTATUS: NO CONSISTENT SYSTEMS FOUND\n\n")
                continue

            max_delta = 1.0
            best_sys = None
            best_iis = None
            all_strictly_metrizable = True

            # בדיקת המערכות עם Progress Bar
            for sys in tqdm(systems, desc=f"Checking Deltas for {part_str}", unit="sys"):
                delta, iis = get_system_delta(G, sys)

                if delta > 1.0:
                    all_strictly_metrizable = False

                if delta > max_delta:
                    max_delta = delta
                    best_sys = sys
                    best_iis = iis

            # כתיבת התוצאות לקובץ לפי הלוגיקה המבוקשת
            f.write(f"Partition {part_str}:\n")
            if all_strictly_metrizable:
                f.write("STATUS: STRONGLY METRIZABLE (All systems Delta = 1.0)\n")
            elif max_delta <= 1.001:
                f.write(f"STATUS: NOT METRIZABLE, BUT MAX DELTA ≈ 1.0 ({max_delta:.5f})\n")
                f.write("(Found a balanced contradiction. The system variables can be inflated to solve for any eps > 0)\n")
            else:
                f.write(f"STATUS: HIGH DELTA DETECTED! Max Delta = {max_delta:.5f}\n")
                f.write("System Details (The Hardest Path System):\n")
                for (u, v), path in best_sys.items():
                    if u < v:  # Print only one direction to save space
                        f.write(f"  {u} -> {v}: {path}\n")
                f.write("\nIIS Constraints (The locked contradiction at maximal failing alpha):\n")
                for c in best_iis:
                    f.write(f"  {c}\n")

            f.write("-" * 60 + "\n\n")

    print(f"\nFinished! Results saved to '{output_path}'.")


if __name__ == "__main__":
    main()
