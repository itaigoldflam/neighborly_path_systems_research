from path_systems_lib.graph_families import grid
from path_systems_lib.io_utils import output_dir
from path_systems_lib.metric_lp import check_metric
from path_systems_lib.path_systems import generate_neighborly_systems

# ==========================================
# Example run for 3x3 Standard Grid
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    STRICTLY_METRIZABLE = True  # Set to True for strict inequalities, False for non-strict (weak)
    n, m_dim = 2, 4

    G = grid(n, m_dim)
    metrizability_type = "strictly" if STRICTLY_METRIZABLE else "non_strictly"
    output_path = output_dir(__file__) / f"standard_grid_{n}x{m_dim}_{metrizability_type}_results.txt"

    print(f"\n[INIT] Standard Grid {n}x{m_dim}: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges.")

    # Using length_tolerance to find alternative (not just shortest-path) systems.
    # Note: Bipartite graphs require tolerance to be even (e.g., 2) to find alternative longer paths.
    print(f"[START] Searching for up to 5000 systems with tolerance 3.")
    systems = generate_neighborly_systems(G, length_tolerance=3, max_systems=5000, order="shortest_first")
    print(f"\n[INFO] Generated {len(systems)} valid path systems. Analyzing metrizability...")

    non_metrizable_count = 0
    metrizable_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, paths in enumerate(systems):
            result = check_metric(paths, strict=STRICTLY_METRIZABLE, compute_iis=True)

            if result.feasible:
                print(f"System {idx} is METRIZABLE (Skipping file output)")
                metrizable_count += 1
            else:
                non_metrizable_count += 1
                print(f"System {idx} is NOT METRIZABLE. Writing details to file...")

                f.write(f"{'=' * 60}\n")
                f.write(f"SYSTEM {idx} - NON METRIZABLE\n")
                f.write(f"{'=' * 60}\n\n")

                f.write("--- PATH SYSTEM ---\n")
                for pair, path in sorted(paths.items()):
                    if pair[0] < pair[1]:
                        f.write(f"  {pair[0]} -> {pair[1]}: {path}\n")

                f.write("\n--- THE CONTRADICTION (IIS) ---\n")
                f.write("Gurobi found that the following subset of constraints is impossible to satisfy simultaneously:\n")
                for c in result.iis:
                    f.write(f"  {c}\n")

                f.write("\n--- FULL SYSTEM OF INEQUALITIES ---\n")
                for c in result.all_constraints:
                    f.write(f"  {c}\n")

                f.write("\n\n")

    print(f"\n[DONE] Metrizable: {metrizable_count}, Non-Metrizable: {non_metrizable_count}.")
    if non_metrizable_count > 0:
        print(f"Detailed contradictions saved to '{output_path}'.")
