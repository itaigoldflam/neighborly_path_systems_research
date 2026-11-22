from typing import List, Tuple

from path_systems_lib.theta_ab2 import extract_parameters, get_consistent_systems


def verify_theta_characterization(R: List[int], L: List[int], k_A: int, k_B: int, a: int, b: int) -> Tuple[bool, List[str]]:
    """ Verifies that the system satisfies all algebraic characterization rules. """
    N = a + b
    errors = []

    # 1. Basic rules (Monotonicity, Symmetry, Bounds)
    for i in range(N):
        if R[i] < 1 or L[i] < 1:
            errors.append(f"R or L < 1 at index {i}")
        if R[i] + L[i] > N - 1:
            errors.append(f"Overlap too big: R+L > N-1 at index {i}")
        if L[(i + R[i]) % N] < R[i]:
            errors.append(f"L/R Symmetry failed (Right) at {i}")
        if R[(i - L[i]) % N] < L[i]:
            errors.append(f"L/R Symmetry failed (Left) at {i}")
        if R[(i + 1) % N] < R[i] - 1:
            errors.append(f"R Monotonicity failed at {i}")
        if L[(i - 1) % N] < L[i] - 1:
            errors.append(f"L Monotonicity failed at {i}")

    # 2. Pure regions (Bottleneck equations)
    for x in range(1, a - L[a]):
        if (x - L[x]) % N != (0 - L[0]) % N:
            errors.append(f"A_v* bottleneck failed at x={x}")

    for x in range(R[0] + 1, a):
        if (x + R[x]) % N != (a + R[a]) % N:
            errors.append(f"A_u* bottleneck failed at x={x}")

    for x in range(a + 1, N - L[0]):
        if (x - L[x]) % N != (a - L[a]) % N:
            errors.append(f"B_u* bottleneck failed at x={x}")

    for x in range(a + R[a] + 1, N):
        if (x + R[x]) % N != (0 + R[0]) % N:
            errors.append(f"B_v* bottleneck failed at x={x}")

    # 3. Intersection regions (Closed cycle equation)
    for x in range(a - L[a], R[0] + 1):
        if (x + R[x] + 1) % N != (x - L[x]) % N:
            errors.append(f"I_A closure failed at x={x}")

    for x in range(N - L[0], a + R[a] + 1):
        if (x + R[x] + 1) % N != (x - L[x]) % N:
            errors.append(f"I_B closure failed at x={x}")

    # 4. Spanning tree bounds for the center
    if not (a - L[a] - 1 <= k_A <= R[0]):
        errors.append(f"k_A={k_A} out of bounds [{a - L[a] - 1}, {R[0]}]")

    if not (N - L[0] - 1 <= k_B <= a + R[a]):
        errors.append(f"k_B={k_B} out of bounds [{N - L[0] - 1}, {a + R[a]}]")

    return len(errors) == 0, errors


def main():
    a, b = 3, 4  # Can be changed to test asymmetry
    print(f"Generating valid systems for Theta({a}, {b}, 2)...")
    systems, _ = get_consistent_systems(a, b)
    print(f"Found {len(systems)} consistent systems where u-v path uses the center.\n")

    success_count = 0
    for idx, sys in enumerate(systems):
        R, L, k_A, k_B = extract_parameters(sys, a, b)

        # Print current system being evaluated
        print(f"--- Checking System {idx + 1}/{len(systems)} ---")
        print(f"Extracted: R={R} | L={L} | k_A={k_A} | k_B={k_B}")

        is_valid, errors = verify_theta_characterization(R, L, k_A, k_B, a, b)
        if is_valid:
            print("Status: SUCCESS (Matches mathematical characterization)\n")
            success_count += 1
        else:
            print("Status: FAILED!")
            for err in errors:
                print("  -", err)
            print("\n")

    print(f"Verification successful for {success_count}/{len(systems)} systems!")


if __name__ == "__main__":
    main()
