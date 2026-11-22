import time
from typing import Set, Tuple

from path_systems_lib.r_l_vectors import get_L_from_R
from path_systems_lib.theta_ab2 import extract_parameters, get_consistent_systems

# =====================================================================
# Theoretical Generation of Valid R, L, k_A, k_B
# =====================================================================


def generate_monotonic_R(N: int):
    """ Yields all vectors of length N in [1, N-1] satisfying R[(i+1)%N] >= R[i] - 1 """
    def gen_R(curr_R):
        if len(curr_R) == N:
            if curr_R[0] >= curr_R[-1] - 1:
                yield tuple(curr_R)
            return
        last = curr_R[-1]
        for nxt in range(max(1, last - 1), N):
            yield from gen_R(curr_R + [nxt])

    for first in range(1, N):
        yield from gen_R([first])


def verify_algebraic_rules(R: Tuple[int, ...], L: Tuple[int, ...], a: int, b: int) -> bool:
    """ Validates if R and L satisfy the characterization logic. """
    N = a + b

    # Endpoints cannot reach each other directly over the rims
    if R[0] >= a or L[a] >= a:
        return False
    if R[a] >= b or L[0] >= b:
        return False

    # Intersection overlaps must exist
    if R[0] + L[a] < a - 1:
        return False
    if R[a] + L[0] < b - 1:
        return False

    # 1. Symmetry and Overlap bounds
    for i in range(N):
        if R[i] + L[i] > N - 1:
            return False
        if L[(i + R[i]) % N] < R[i]:
            return False
        if R[(i - L[i]) % N] < L[i]:
            return False

    # 2. Pure Regions (Bottleneck equalities)
    for x in range(1, a - L[a]):
        idx = x % N
        if (x - L[idx]) % N != (0 - L[0]) % N:
            return False

    for x in range(R[0] + 1, a):
        idx = x % N
        if (x + R[idx]) % N != (a + R[a]) % N:
            return False

    for x in range(a + 1, N - L[0]):
        idx = x % N
        if (x - L[idx]) % N != (a - L[a]) % N:
            return False

    for x in range(a + R[a] + 1, N):
        idx = x % N
        if (x + R[idx]) % N != (0 + R[0]) % N:
            return False

    # 3. Intersection regions (Cycle closures)
    for x in range(a - L[a], R[0] + 1):
        idx = x % N
        if (x + R[idx] + 1) % N != (x - L[idx]) % N:
            return False

    for x in range(N - L[0], a + R[a] + 1):
        idx = x % N
        if (x + R[idx] + 1) % N != (x - L[idx]) % N:
            return False

    return True


def get_all_theoretical_parameters(a: int, b: int) -> Set[Tuple[Tuple[int, ...], Tuple[int, ...], int, int]]:
    """ Generates the set of all theoretically valid (R, L, k_A, k_B) tuples using R-derivation. """
    N = a + b
    valid_params = set()

    for R in generate_monotonic_R(N):
        # Quick filter
        if R[0] >= a or R[a] >= b:
            continue

        # Check pure R regions early to save time
        r_valid = True
        for x in range(R[0] + 1, a):
            if (x + R[x % N]) % N != (a + R[a]) % N:
                r_valid = False
                break
        if not r_valid:
            continue

        for x in range(a + R[a] + 1, N):
            if (x + R[x % N]) % N != (0 + R[0]) % N:
                r_valid = False
                break
        if not r_valid:
            continue

        # Derive L uniquely from R
        L = tuple(get_L_from_R(list(R), N))

        if verify_algebraic_rules(R, L, a, b):
            min_ka = max(0, a - L[a] - 1)
            max_ka = min(a, R[0])
            min_kb = max(a, N - L[0] - 1)
            max_kb = min(N, a + R[a])

            for kA in range(min_ka, max_ka + 1):
                for kB in range(min_kb, max_kb + 1):
                    valid_params.add((R, L, kA, kB))

    return valid_params


# =====================================================================
# Main Execution and Comparison
# =====================================================================

def main():
    a, b = 6, 5

    print(f"--- 1. Generating actual consistent systems for Theta({a}, {b}, 2) ---")
    start_time_1 = time.time()
    actual_systems, _ = get_consistent_systems(a, b)

    actual_params_set = set()
    for sys in actual_systems:
        R, L, k_A, k_B = extract_parameters(sys, a, b)
        actual_params_set.add((tuple(R), tuple(L), k_A, k_B))
    end_time_1 = time.time()

    print(f"Total actual path systems found: {len(actual_systems)} (took {end_time_1 - start_time_1:.4f}s)")
    print(f"Total unique (R, L, k_A, k_B) from actual systems: {len(actual_params_set)}\n")

    print("--- 2. Generating theoretical parameters using algebraic model ---")
    start_time_2 = time.time()
    theoretical_params_set = get_all_theoretical_parameters(a, b)
    end_time_2 = time.time()
    print(f"Total theoretically valid (R, L, k_A, k_B) found: {len(theoretical_params_set)} (took {end_time_2 - start_time_2:.4f}s)\n")

    print("--- 3. Comparison Results ---")
    if actual_params_set == theoretical_params_set:
        print("SUCCESS! PERFECT MATCH (Bijection Achieved)!")
    else:
        print("MISMATCH FOUND!")


if __name__ == "__main__":
    main()
