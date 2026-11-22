from typing import Dict, Literal

from path_systems_lib.graph_families import wheel
from path_systems_lib.metric_lp import optimal_metric_value
from path_systems_lib.path_systems import get_neighborly_path_systems_exact

ObjectiveKind = Literal["sum", "max"]


def wheel_neighborly_metrizability_with_objective(
    n: int,
    objective: ObjectiveKind = "sum",
    strict: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    For every neighborly path system of the n-wheel, compute the optimal value of
    the chosen objective (sum or max edge). Returns the worst-case objective
    value across all systems and the index of a system attaining it.
    """
    if n < 4:
        raise ValueError("Wheel requires at least 4 vertices.")

    G = wheel(n)
    all_systems = get_neighborly_path_systems_exact(G)

    hub = 0
    rim = [v for v in range(n) if v != hub]
    wheel_edges = [(hub, v) for v in rim] + [(rim[i], rim[(i + 1) % len(rim)]) for i in range(len(rim))]

    worst_value = float("-inf")
    worst_idx = -1
    worst_distances = None

    for idx, p_system in enumerate(all_systems):
        result = optimal_metric_value(p_system, wheel_edges, objective=objective, strict=strict)
        if result is None:
            # Infeasible system; skip
            continue
        value, distances = result
        if verbose:
            print(f"[system {idx}] objective={value}")
        if value > worst_value:
            worst_value = value
            worst_idx = idx
            worst_distances = distances

    return {
        "n": n,
        "objective": objective,
        "strict": strict,
        "systems_evaluated": len(all_systems),
        "worst_value": worst_value,
        "worst_system_index": worst_idx,
        "worst_distances": worst_distances,
        "wheel_edges": wheel_edges,
        "worst_p_system": all_systems[worst_idx],
    }


def main():
    # Example usage: change parameters as desired.
    for n in range(4, 8):
        summary = wheel_neighborly_metrizability_with_objective(
            n=n,
            objective="sum",  # options: "sum", "max"
            strict=True,
            verbose=False,
        )
        print(f"n={n}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    main()
