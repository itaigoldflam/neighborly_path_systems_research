from path_systems_lib.graph_families import grid
from path_systems_lib.io_utils import output_dir
from path_systems_lib.metric_lp import check_metric
from path_systems_lib.path_systems import generate_neighborly_systems
from path_systems_lib.viz import draw_grid_system, get_maximal_paths

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    n, m_dim = 3, 3
    G = grid(n, m_dim)

    output_folder = output_dir(__file__) / "grid_results"
    output_folder.mkdir(parents=True, exist_ok=True)
    text_output_path = output_folder / f"standard_grid_{n}x{m_dim}_results.txt"

    print(f"\n[INIT] Standard Grid {n}x{m_dim}: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges.")

    # Note: Bipartite grids need tolerance=2 to find longer paths and potential contradictions
    print(f"[START] Searching for up to 5000 systems with tolerance 3.")
    systems = generate_neighborly_systems(G, length_tolerance=3, max_systems=5000, order="shortest_first")
    print(f"\n[INFO] Generated {len(systems)} valid path systems. Analyzing metrizability...")

    non_metrizable_count = 0
    metrizable_count = 0

    with open(text_output_path, "w", encoding="utf-8") as f:
        for idx, paths in enumerate(systems):
            result = check_metric(paths, strict=True, compute_iis=True)

            if result.feasible:
                print(f"System {idx} is METRIZABLE")
                metrizable_count += 1
            else:
                non_metrizable_count += 1
                print(f"System {idx} is NOT METRIZABLE. Writing details and drawing graph...")

                f.write(f"{'=' * 60}\n")
                f.write(f"SYSTEM {idx} - NON METRIZABLE\n")
                f.write(f"{'=' * 60}\n\n")

                f.write("--- THE CONTRADICTION (IIS) ---\n")
                for c in result.iis:
                    f.write(f"  {c}\n")
                f.write("\n\n")

                maximal_paths = get_maximal_paths(paths)
                img_path = output_folder / f"system_{idx}_graph.png"
                draw_grid_system(n, m_dim, maximal_paths, img_path)

    print(f"\n[DONE] Metrizable: {metrizable_count}, Non-Metrizable: {non_metrizable_count}.")
    if non_metrizable_count > 0:
        print(f"Details saved to {text_output_path}")
        print(f"Images saved to the '{output_folder}' directory.")
