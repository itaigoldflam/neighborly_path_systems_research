from itertools import combinations_with_replacement

from tqdm import tqdm

from path_systems_lib.graph_families import theta_general
from path_systems_lib.io_utils import output_dir
from path_systems_lib.path_systems import generate_neighborly_systems
from path_systems_lib.path_weight_lp import build_alpha_model


def format_path(path):
    return "-".join(map(str, path))


# --- Main Execution Block ---
if __name__ == "__main__":
    MAX_BRANCH_LENGTH = 4  # You can change the desired maximum here
    STRONG_METRIZABILITY = True  # Change to False for weak metrizability

    metrizability_type = "Strong" if STRONG_METRIZABILITY else "Weak"
    epsilon = 0.001

    results_dir = output_dir(__file__) / "Theta_Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing Theta(a, b, c) graphs for 1 <= a <= b <= c <= {MAX_BRANCH_LENGTH} ({metrizability_type} Metrizability)\n")

    # The loop now runs on triplets (a, b, c) such that a <= b <= c
    for a, b, c in combinations_with_replacement(range(1, MAX_BRANCH_LENGTH + 1), 3):
        if (a, b, c) != (2, 3, 4):
            continue

        print(f"--- Building Theta({a}, {b}, {c}) ---")
        G = theta_general(a, b, c)

        systems = generate_neighborly_systems(G, length_tolerance=None, max_systems=None, order="mrv")

        non_metrizable_systems = []
        metrizable_count = 0

        for system in tqdm(systems, desc=f"Solving LP ({metrizability_type}) for Theta({a},{b},{c})", unit="sys"):
            result = build_alpha_model(G, system, alpha=1.0, eps=epsilon if STRONG_METRIZABILITY else 0.0, compute_iis=True)
            if result.feasible:
                metrizable_count += 1
            else:
                non_metrizable_systems.append((system, result.iis or []))

        print(f"Results: {metrizable_count} {metrizability_type.lower()} metrizable out of {len(systems)} total.\n")

        if non_metrizable_systems:
            filename = results_dir / f"{metrizability_type}_NonMetrizable_Theta_{a}_{b}_{c}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Graph: Theta({a}, {b}, {c})\n")
                f.write(f"Metrizability: {metrizability_type}\n")
                f.write(f"Total Consistent Systems: {len(systems)}\n")
                f.write(f"Non-Metrizable Systems: {len(non_metrizable_systems)}\n")
                f.write("=" * 50 + "\n\n")

                for idx, (sys, contra) in enumerate(non_metrizable_systems, 1):
                    f.write(f"--- System #{idx} ---\n")
                    f.write("Chosen Paths:\n")
                    for pair in sorted(sys.keys()):
                        f.write(f"  {pair}: {format_path(sys[pair])}\n")

                    f.write("\nContradiction (IIS Constraints):\n")
                    for eq in contra:
                        f.write(f"  {eq}\n")
                    f.write("-" * 50 + "\n\n")
            print(f"Saved {len(non_metrizable_systems)} non-metrizable systems to {filename}\n")
