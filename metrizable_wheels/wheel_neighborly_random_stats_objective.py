import argparse
import random
from typing import Generator, List, Optional, Tuple, Literal

from path_systems_lib.metric_lp import is_system_metric, optimal_metric_value
from path_systems_lib.path_tree_systems import InconsistentPathError, PathSystem
from path_systems_lib.r_l_vectors import is_legal_neighborly_vector as _is_legal_neighborly_vectors

VectorPair = Tuple[List[int], List[int]]
ObjectiveKind = Literal["sum", "max"]

# -----------------------------------------------------------------------------
#  LP Logic (Added for Objective Calculation)
# -----------------------------------------------------------------------------

def optimal_metric_value_for_system(
    p_system,
    n_rim: int,
    objective: ObjectiveKind = "sum",
    strict: bool = True,
) -> Optional[float]:
    # In this script's convention, Rim is 0..n-1, Hub is n.
    hub = n_rim
    rim = list(range(n_rim))
    wheel_edges = [(hub, v) for v in rim] + [(rim[i], rim[(i + 1) % len(rim)]) for i in range(len(rim))]

    result = optimal_metric_value(p_system, wheel_edges, objective=objective, strict=strict)
    return result[0] if result is not None else None

# -----------------------------------------------------------------------------
#  Original Random Walk Logic
# -----------------------------------------------------------------------------


def _random_neighborly_walk(
    n: int,
    samples: int,
    walk_length: int,
    seed: Optional[int] = None,
    max_attempt_factor: int = 40,
    require_unique: bool = True,
    seen: Optional[set] = None,
    max_unique_rejections: int = 10000,
) -> Generator[VectorPair, None, None]:
    """
    Yield `samples` neighborly vectors via a lazy random walk that preserves legality.
    """
    rng = random.Random(seed)
    r = [1] * n
    l = [1] * n
    assert _is_legal_neighborly_vectors(r, l)
    if seen is None:
        seen = set()

    def idx(i: int) -> int:
        return i % n

    def dec_r(i: int) -> None:
        # print("dec_r", i)
        i = idx(i)
        if r[i] == 1:
            return
        if r[idx(i - 1)] == r[i] + 1:
            dec_r(i - 1)
        r[i] -= 1
        guard = 2 * n
        while l[idx(i + r[i] + 1)] > r[i]:
            if guard <= 0:
                return
            guard -= 1
            dec_l(i + r[i] + 1)

    def dec_l(i: int) -> None:
        # print("dec_l", i)
        i = idx(i)
        if l[i] == 1:
            return
        if l[idx(i + 1)] == l[i] + 1:
            dec_l(i + 1)
        l[i] -= 1
        guard = 2 * n
        while r[idx(i - l[i] - 1)] > l[i]:
            if guard <= 0:
                return
            guard -= 1
            dec_r(i - l[i] - 1)

    def inc_r(i: int) -> None:
        # print("inc_r", i)
        i = idx(i)
        if r[i] + l[i] == n - 1:
            dec_l(i)
        if r[idx(i + 1)] == r[i] - 1:
            inc_r(i + 1)
        r[i] += 1
        guard = 2 * n
        while l[idx(i + r[i])] < r[i]:
            if guard <= 0:
                return
            guard -= 1
            inc_l(i + r[i])

    def inc_l(i: int) -> None:
        # print("inc_l", i)
        i = idx(i)
        if r[i] + l[i] == n - 1:
            dec_r(i)
        if l[idx(i - 1)] == l[i] - 1:
            inc_l(i - 1)
        l[i] += 1
        guard = 2 * n
        while r[idx(i - l[i])] < l[i]:
            if guard <= 0:
                return
            guard -= 1
            inc_r(i - l[i])

    produced = 0
    duplicate_rejections = 0
    while produced < samples and duplicate_rejections < max_unique_rejections:
        accepted = 0
        attempts = 0
        max_attempts = max_attempt_factor * max(1, walk_length)
        while accepted < walk_length and attempts < max_attempts:
            attempts += 1
            which = rng.choice(("r", "l"))
            direction = rng.choice(("inc", "dec"))
            i_choice = rng.randrange(n)

            old_r = list(r)
            old_l = list(l)

            if which == "r":
                if direction == "inc":
                    inc_r(i_choice)
                else:
                    dec_r(i_choice)
            else:
                if direction == "inc":
                    inc_l(i_choice)
                else:
                    dec_l(i_choice)

            if (r, l) == (old_r, old_l):
                continue

            if not _is_legal_neighborly_vectors(r, l):
                print("illegal state", r, l)
                # Revert on illegal state.
                r = old_r
                l = old_l
                continue

            accepted += 1

        key = (tuple(r), tuple(l))
        if require_unique and key in seen:
            duplicate_rejections += 1
            continue

        seen.add(key)
        produced += 1
        yield list(r), list(l)


def vectors_to_wheel_path_system(
    r: List[int],
    l: List[int],
    hub: Optional[int] = None,
) -> PathSystem:
    """
    Translate neighborly (R, L) vectors into a PathSystem on a wheel graph.

    Rim vertices are numbered 0..n-1 (indices modulo n). The hub defaults to n so
    that its label does not collide with rim indices.
    """
    n = len(r)
    if len(l) != n:
        raise ValueError("R and L must have the same length.")
    hub_id = n if hub is None else hub
    system = PathSystem(consistent=True)

    # Hub-to-rim spokes.
    for v in range(n):
        system[(hub_id, v)] = (hub_id, v)

    # Rim paths following the right (clockwise) and left (counter-clockwise) directions.
    for i in range(n):
        for step in range(1, r[i] + 1):
            path = [(i + offset) % n for offset in range(step + 1)]
            system[(path[0], path[-1])] = tuple(path)
        for step in range(1, l[i] + 1):
            path = [(i - offset) % n for offset in range(step + 1)]
            system[(path[0], path[-1])] = tuple(path)
    return system


def run_statistical_wheel_metrizability_check(
    n: int,
    samples: int = 100,
    walk_length: Optional[int] = None,
    seed: Optional[int] = None,
    strict: bool = True,
    verbose_every: int = 0,
    log_each: bool = False,
    require_unique: bool = True,
    max_unique_rejections: int = 10000,
    objective: Optional[ObjectiveKind] = None,
) -> dict:
    """
    Generate random neighborly vectors and estimate strict metrizability frequency.

    Returns a summary dictionary with counts and estimated proportion.
    """
    if n < 4:
        raise ValueError("Wheel requires at least 4 rim vertices.")
    walk = walk_length or max(10, 5 * n)
    metrics = 0
    built = 0
    failures = 0
    
    # New Stats for Objective
    worst_value = float("-inf")
    worst_vectors = None

    seen = set()

    for idx, (r, l) in enumerate(
        _random_neighborly_walk(
            n,
            samples,
            walk,
            seed=seed,
            require_unique=require_unique,
            seen=seen,
            max_unique_rejections=max_unique_rejections,
        ),
        start=1,
    ):
        if log_each:
            print(f"[sample {idx}] legal vector: R={r}, L={l}")
        try:
            system = vectors_to_wheel_path_system(r, l)
        except InconsistentPathError:
            failures += 1
            continue

        if log_each:
            print(f"[sample {idx}] running metrizability test...")
        
        # Modified Logic: check objective if requested, otherwise standard check
        val = None
        if objective:
            val = optimal_metric_value_for_system(system, n, objective=objective, strict=strict)
            is_metric = (val is not None)
            print(f"val: {val}")
            if is_metric and val > worst_value:
                worst_value = val
                worst_vectors = (r, l)
        else:
            distances = is_system_metric(system, strict=strict)
            is_metric = distances is not None

        metrics += int(is_metric)
        built += 1

        if verbose_every and idx % verbose_every == 0:
            msg = f"sample {idx}: metric={is_metric}, R={r}, L={l}"
            if objective and is_metric:
                msg += f" | {objective}={val:.4f}"
            print(msg)

    evaluated = built
    proportion = metrics / evaluated if evaluated else 0.0
    
    summary = {
        "n": n,
        "samples_requested": samples,
        "samples_evaluated": evaluated,
        "metric_count": metrics,
        "failed_builds": failures,
        "metric_fraction": proportion,
    }
    
    if objective:
        summary["objective"] = objective
        summary["worst_value"] = worst_value if metrics > 0 else None
        summary["worst_vectors"] = worst_vectors if metrics > 0 else None

    return summary


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-walk statistical metrizability test for wheel graphs.")
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of rim vertices.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="How many random vectors to test (default 200).",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=None,
        help="Number of accepted walk steps between samples (default 5n).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require strict metrizability (default).",
    )
    parser.add_argument(
        "--non-strict",
        dest="strict",
        action="store_false",
        help="Allow non-strict metrizability.",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=0,
        help="Print every k samples (0 to disable).",
    )
    # Added objective argument
    parser.add_argument(
        "--objective",
        type=str,
        choices=["sum", "max"],
        default=None,
        help="Optimize 'sum' or 'max' edge length. Default is None (feasibility only).",
    )
    return parser.parse_args()


def main(
    n: Optional[int] = None,
    samples: int = 200,
    walk_length: Optional[int] = None,
    seed: Optional[int] = None,
    strict: bool = True,
    verbose_every: int = 0,
    use_cli_args: bool = False,
    log_each: bool = False,
    require_unique: bool = True,
    max_unique_rejections: int = 10000,
    objective: Optional[ObjectiveKind] = None, # Added param
) -> dict:
    """
    Entry point. By default uses the provided arguments; set use_cli_args=True to read CLI.
    """
    if use_cli_args:
        args = _parse_cli()
        n = args.n
        samples = args.samples
        walk_length = args.walk_length
        seed = args.seed
        strict = args.strict
        verbose_every = args.verbose_every
        objective = args.objective # Read from args
        # If CLI is used, keep log_each as passed to main (no extra flag available).

    summary = run_statistical_wheel_metrizability_check(
        n=n,
        samples=samples,
        walk_length=walk_length,
        seed=seed,
        strict=strict,
        verbose_every=verbose_every,
        log_each=log_each,
        require_unique=require_unique,
        max_unique_rejections=max_unique_rejections,
        objective=objective, # Pass to runner
    )
    if use_cli_args:
        print(summary)
    return summary


if __name__ == "__main__":
    # Set your parameters here (no need to pass CLI args).
    summary = main(
        n=100,
        samples=200,
        walk_length=2000,  # accepted random moves between checks (defaults to 5n)
        seed=None,         # set an int for reproducibility
        strict=True,       # set False to allow non-strict
        verbose_every=0,   # >0 to print every k samples
        use_cli_args=False,
        log_each=True,     # True to print every legal vector and test
        require_unique=True,
        max_unique_rejections=10000,
        objective="max"     # "sum" or "max"
    )
    print(summary)