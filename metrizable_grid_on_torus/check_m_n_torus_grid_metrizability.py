from path_systems_lib.graph_families import torus_grid
from path_systems_lib.io_utils import output_dir
from path_systems_lib.metric_lp import check_metric
from path_systems_lib.path_systems import generate_neighborly_systems

# ==========================================
# Example run for 3x3 Torus
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    STRICTLY_METRIZABLE = False  # Set to True for strict inequalities, False for non-strict (weak)
    n, m_dim = 12, 2

    G = torus_grid(n, m_dim)
    metrizability_type = "strictly" if STRICTLY_METRIZABLE else "non_strictly"
    output_path = output_dir(__file__) / f"torus_{n}x{m_dim}_{metrizability_type}_non_metrizable_details.txt"

    # Notice length_tolerance=2 to allow paths of length 3, creating potential contradictions!
    print(f"[START] Searching for up to 100 systems with tolerance 2.")
    systems = generate_neighborly_systems(G, length_tolerance=2, max_systems=100, order="shortest_first")
    print(f"\n[INFO] Generated {len(systems)} valid path systems. Analyzing metrizability...")

    non_metrizable_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, paths in enumerate(systems):
            result = check_metric(paths, strict=STRICTLY_METRIZABLE, compute_iis=True)

            if result.feasible:
                print(f"System {idx} is METRIZABLE (Skipping file output)")
            else:
                non_metrizable_count += 1
                print(f"System {idx} is NOT METRIZABLE. Writing details to file...")

                f.write("=" * 60 + "\n")
                f.write(f"SYSTEM {idx} - NON METRIZABLE\n")
                f.write("=" * 60 + "\n\n")

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

    print(f"\n[DONE] Found {non_metrizable_count} non-metrizable systems.")
    print(f"Detailed contradictions saved to '{output_path}'.")
