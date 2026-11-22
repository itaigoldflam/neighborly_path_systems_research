from path_systems_lib.graph_families import get_unique_spoke_masks, mask_to_partition_string, partial_wheel
from path_systems_lib.io_utils import output_dir
from path_systems_lib.metric_lp import check_metric
from path_systems_lib.path_systems import generate_neighborly_systems

# ==========================================
# Main Orchestration
# ==========================================


def run_experiment_for_n(n: int, max_systems_per_partition: float = float("inf"), exhaustive_paths: bool = True):
    unique_masks = get_unique_spoke_masks(n)
    output_path = output_dir(__file__) / f"partial_wheels_n_{n}_results.txt"

    print(f"\n[INIT] Starting experiment for n={n}")
    print(f"[INIT] Found {len(unique_masks)} unique partial wheel partitions.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"EXPERIMENT REPORT FOR n={n}\n")
        f.write(f"Total unique partitions: {len(unique_masks)}\n")
        f.write("=" * 60 + "\n\n")

        for mask_idx, mask in enumerate(unique_masks):
            partition_str = mask_to_partition_string(mask)
            G = partial_wheel(n, mask)

            print(f"[{mask_idx + 1}/{len(unique_masks)}] Analyzing Partition: {partition_str} (Mask: {mask})")
            f.write(f"--- Partition {mask_idx + 1}: {partition_str} ---\n")
            f.write(f"Spoke Mask: {mask}\n")
            f.write(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges\n")

            # Generate ALL systems natively. exhaustive_paths=True -> no length cutoff
            # at all; False -> restrict every pair to its shortest-path length only
            # (length_tolerance=0), matching the original's two modes exactly.
            length_tolerance = None if exhaustive_paths else 0
            max_systems = None if max_systems_per_partition == float("inf") else int(max_systems_per_partition)
            systems = generate_neighborly_systems(
                G,
                length_tolerance=length_tolerance,
                max_systems=max_systems,
                order="shortest_first",
            )
            f.write(f"Generated {len(systems)} valid neighborly systems.\n\n")

            if not systems:
                f.write("No consistent systems found.\n\n")
                continue

            partition_is_strong = True
            partition_is_weak = True
            strong_counterexample = None
            weak_counterexample = None

            for sys_idx, paths in enumerate(systems):
                strong_result = check_metric(paths, strict=True, compute_iis=True)
                if not strong_result.feasible:
                    partition_is_strong = False
                    if strong_counterexample is None:
                        strong_counterexample = (sys_idx, strong_result.iis)

                    weak_result = check_metric(paths, strict=False, compute_iis=True)
                    if not weak_result.feasible:
                        partition_is_weak = False
                        if weak_counterexample is None:
                            weak_counterexample = (sys_idx, weak_result.iis)
                        break

            if partition_is_strong:
                f.write("STATUS: STRONGLY METRIZABLE (All systems are strictly feasible)\n")
            elif partition_is_weak and not partition_is_strong:
                f.write("STATUS: WEAKLY BUT NOT STRONGLY METRIZABLE\n")
                f.write("A counterexample for STRONG metrizability was found:\n")
                sys_idx, iis_str = strong_counterexample
                f.write(f"System Index: {sys_idx}\n")
                f.write("IIS Constraints (The Contradiction in Strong):\n")
                for c in iis_str:
                    f.write(f"  {c}\n")
            elif not partition_is_weak:
                f.write("STATUS: NOT WEAKLY METRIZABLE\n")
                f.write("A counterexample for WEAK metrizability was found (discarding strong counterexample):\n")
                sys_idx, iis_wk = weak_counterexample
                f.write(f"System Index: {sys_idx}\n")
                f.write("IIS Constraints (The Contradiction in Weak):\n")
                for c in iis_wk:
                    f.write(f"  {c}\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"\n[DONE] Finished n={n}. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment_for_n(7)
