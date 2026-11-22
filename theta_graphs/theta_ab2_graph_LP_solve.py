from tqdm import tqdm

from path_systems_lib.io_utils import output_dir
from path_systems_lib.path_weight_lp import build_alpha_model
from path_systems_lib.theta_ab2 import extract_parameters, get_consistent_systems

# =====================================================================
# Main Execution and File Writing
# =====================================================================


def solve_system_weights_gurobi(system, G, a, b, strong=True):
    """Find the optimal (minimal total-weight) strict weights for `system` via
    path_weight_lp. epsilon=1.0 for strong metrizability, 0.0 for weak."""
    N = a + b
    center = N
    result = build_alpha_model(G, system, alpha=1.0, eps=1.0 if strong else 0.0, objective="sum")
    if not result.feasible:
        return False, [], 0.0, 0.0

    rim_weights = [round(result.weights[tuple(sorted((i, (i + 1) % N)))], 3) for i in range(N)]
    w_v = round(result.weights[tuple(sorted((0, center)))], 3)
    w_u = round(result.weights[tuple(sorted((a, center)))], 3)
    return True, rim_weights, w_v, w_u


def main():
    a, b = 4, 3
    STRONG = False  # Toggle this for strong or weak metrizability
    print(f"Generating all consistent systems for Theta({a}, {b}, 2)...")
    systems, G = get_consistent_systems(a, b)

    metrizability_type = "Strong" if STRONG else "Weak"
    print(f"Found {len(systems)} systems. Solving LPs via Gurobi ({metrizability_type} Metrizability)...")

    output_path = output_dir(__file__) / f"Theta_{a}_{b}_LP_Weights_{metrizability_type}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"LP Weight Results for Theta({a}, {b}, 2) - {metrizability_type}\n")
        f.write(f"Total Systems: {len(systems)}\n")
        f.write("=" * 60 + "\n\n")

        for idx, sys in enumerate(tqdm(systems, desc=f"Solving LPs ({metrizability_type})")):
            R, L, k_A, k_B = extract_parameters(sys, a, b)
            is_metrizable, rim_w, w_v, w_u = solve_system_weights_gurobi(sys, G, a, b, strong=STRONG)

            f.write(f"--- System #{idx + 1} ---\n")
            f.write(f"R:  {R}\n")
            f.write(f"L:  {L}\n")
            f.write(f"kA: {k_A} | kB: {k_B}\n")

            if is_metrizable:
                f.write("Weights:\n")
                f.write(f"  Rim:    {rim_w}\n")
                f.write(f"  Spokes: v={w_v}, u={w_u}\n")
            else:
                f.write("Weights: FAILED LP (Not Metrizable)\n")
            f.write("\n")

    print(f"\nDone! Results saved to '{output_path}'.")


if __name__ == "__main__":
    main()
