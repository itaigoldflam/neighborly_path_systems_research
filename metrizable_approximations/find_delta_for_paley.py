import networkx as nx
from typing import List, Tuple, Dict

from path_systems_lib.delta_search import find_delta
from path_systems_lib.graph_families import paley
from path_systems_lib.io_utils import output_dir
from path_systems_lib.path_weight_lp import build_alpha_model

# ==========================================
# Consistent Path System Generator
# ==========================================


def is_consistent(new_path: List[int], current_paths: Dict[Tuple[int, int], List[int]]) -> bool:
    length = len(new_path)
    for i in range(length):
        for j in range(i + 1, length):
            sub = new_path[i:j + 1]
            u, v = sub[0], sub[-1]
            if (u, v) in current_paths and current_paths[(u, v)] != sub:
                return False
            if (v, u) in current_paths and current_paths[(v, u)] != sub[::-1]:
                return False

    new_u, new_v = new_path[0], new_path[-1]
    for _, existing_path in current_paths.items():
        if new_u in existing_path and new_v in existing_path:
            idx_u = existing_path.index(new_u)
            idx_v = existing_path.index(new_v)
            if idx_u < idx_v:
                segment = existing_path[idx_u:idx_v + 1]
            else:
                segment = existing_path[idx_v:idx_u + 1][::-1]
            if segment != new_path and segment != new_path[::-1]:
                return False

    return True


def generate_neighborly_systems(G: nx.Graph, max_solutions: int = 1, cutoff: int = 3) -> List[Dict[Tuple[int, int], List[int]]]:
    """
    Backtracking algorithm to generate consistent neighborly path systems.
    """
    V = list(G.nodes())
    E = list(G.edges())

    current_paths = {}
    for u in V:
        current_paths[(u, u)] = [u]
    for u, v in E:
        current_paths[(u, v)] = [u, v]
        current_paths[(v, u)] = [v, u]

    missing_pairs = []
    for u in V:
        for v in V:
            if u < v and (u, v) not in current_paths:
                missing_pairs.append((u, v))

    missing_pairs.sort(key=lambda pair: nx.shortest_path_length(G, pair[0], pair[1]))
    all_valid_systems = []

    def backtrack(pair_idx):
        if len(all_valid_systems) >= max_solutions:
            return

        if pair_idx == len(missing_pairs):
            all_valid_systems.append(current_paths.copy())
            print(f"[*] Found system {len(all_valid_systems)}/{max_solutions}")
            return

        u, v = missing_pairs[pair_idx]
        potential_paths = list(nx.all_simple_paths(G, u, v, cutoff=cutoff))

        for path in potential_paths:
            if is_consistent(path, current_paths):
                current_paths[(u, v)] = path
                current_paths[(v, u)] = path[::-1]
                backtrack(pair_idx + 1)
                del current_paths[(u, v)]
                del current_paths[(v, u)]

    print(f"Starting backtracking for {len(missing_pairs)} missing pairs (Cutoff={cutoff})...")
    backtrack(0)
    return all_valid_systems


# ==========================================
# Analysis (Family A LP, via path_weight_lp)
# ==========================================

def analyze_path_system(G: nx.Graph, sys_idx: int, defined_paths: Dict[Tuple[int, int], List[int]], file_handle, tol: float = 1e-4) -> float:
    """
    Analyzes a system and writes output to both console and the provided file_handle.
    Returns the calculated Delta(P) so the main loop can collect systems with Delta > 1.
    """
    def out(text: str):
        print(text)
        file_handle.write(text + "\n")

    out("\n=====================================")
    out(f" ANALYZING SYSTEM {sys_idx}")
    out("=====================================")
    out("--- STEP 1: Checking Metrizability ---")

    is_weak = build_alpha_model(G, defined_paths, alpha=1.0, eps=0.0, path_cutoff=4, skip_trivial_edges=True).feasible
    if is_weak:
        is_strong = build_alpha_model(G, defined_paths, alpha=1.0, eps=1e-3, weight_ub=1000.0, path_cutoff=4, skip_trivial_edges=True).feasible
        if is_strong:
            out(">> System IS STRONGLY Metrizable! (Delta(P) = 1.0)")
        else:
            out(">> System is WEAKLY BUT NOT STRONGLY Metrizable. (Delta(P) = 1.0 via infimum)")
        return 1.0

    out("\n--- STEP 2: NOT Weakly Metrizable. Binary Search for Delta(P) ---")

    result = find_delta(G, defined_paths, tol=tol, eps=0.0, path_cutoff=4, skip_trivial_edges=True, auto_expand_hi=False, compute_iis_at_result=False)
    best_alpha = result.delta

    # 'low' represents the maximal alpha for which the system is INFEASIBLE.
    maximal_infeasible = best_alpha - tol
    out(f"\n[+] Minimal Delta(P) ≈ {best_alpha:.5f}")

    # Check if Delta is approximately 1.0 based on the user's logic
    # (i.e. feasible > 1.0, but maximal infeasible <= 1.0)
    if maximal_infeasible <= 1.0 + 1e-6:
        out(">> Delta(P) is approximately 1.0 (Feasible alpha > 1, Infeasible alpha <= 1).")
        out(">> Status only: No strict contradiction (IIS) extracted since Delta ≈ 1.0.")
    else:
        out(f"[!] Extracting Contradiction (IIS) at maximal failing alpha = {maximal_infeasible:.5f}...")
        iis_result = build_alpha_model(G, defined_paths, alpha=maximal_infeasible, eps=0.0, path_cutoff=4, skip_trivial_edges=True, compute_iis=True)

        out("    Conflicting Constraints:")
        for constr in (iis_result.iis or []):
            out(f"      - {constr}")

    return best_alpha


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    p = 13
    print(f"Generating Paley({p}) Graph...")
    G = paley(p)
    print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Let's find a few systems to demonstrate the summary functionality
    max_systems_to_find = 5000
    backtrack_cutoff = 10

    systems = generate_neighborly_systems(G, max_solutions=max_systems_to_find, cutoff=backtrack_cutoff)

    if not systems:
        print(f"Could not find any consistent path systems under cutoff {backtrack_cutoff}.")
    else:
        output_path = output_dir(__file__) / f"paley_{p}_results.txt"
        large_delta_systems = []

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"EXPERIMENT REPORT FOR PALEY GRAPH (p={p})\n")
            f.write("=" * 50 + "\n")

            for i, sys in enumerate(systems):
                delta = analyze_path_system(G, i + 1, sys, f)

                # If Delta > 1.0 strictly (allowing a tiny numerical tolerance)
                if delta > 1.0 + 1e-4:
                    large_delta_systems.append((i + 1, delta))

            # Print and write the final summary list of Delta > 1 systems
            def out_summary(text):
                print(text)
                f.write(text + "\n")

            out_summary("\n\n" + "*" * 60)
            out_summary(" SUMMARY OF SYSTEMS WITH STRICTLY DELTA(P) > 1")
            out_summary("*" * 60)

            if len(large_delta_systems) > 0:
                for idx, d_val in large_delta_systems:
                    out_summary(f" - System #{idx} | Delta(P) ≈ {d_val:.5f}")
            else:
                out_summary(" >> No systems found with Delta(P) strictly > 1.0.")
            out_summary("*" * 60 + "\n")

        print(f"\n[DONE] Execution finished. Results completely saved to {output_path}.")
