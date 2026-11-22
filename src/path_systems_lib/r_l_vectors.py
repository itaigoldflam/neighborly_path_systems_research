"""R/L neighborly-vector toolkit for wheel graphs.

A wheel's rim is an n-cycle (vertices 0..n-1) around a hub. For a "neighborly"
wheel path system, each rim vertex i has a maximal clockwise reach R[i] (the
farthest rim vertex (i+R[i]) % n it connects to directly) and a derived
counter-clockwise reach L[i].

Consolidates ~9-11 copies of this toolkit found across metrizable_wheels/ and its
Reduction_*/Induction/find-not-metrizable subfolders. Two of the five helpers
(`get_L_from_R`, `is_legal_neighborly_vector`) were already identical everywhere
modulo cosmetic renames. The other two were not: see the note on
`get_canonical_bracelet` below for a genuine correctness bug found in roughly half
of the existing copies, and `generate_all_valid_R` below for the three distinct
generation algorithms consolidated into one with a `multiprocess` flag.
"""
import multiprocessing
from typing import List, Optional, Tuple


def ts(u: int, v: int) -> Tuple[int, int]:
    """Sorted pair (min, max). The variadic single-iterable form seen in a couple
    of copies was dead code (every call site in the repo passes two scalars), so
    this keeps the simpler two-argument signature."""
    return tuple(sorted((u, v)))


def get_L_from_R(R: List[int], n: Optional[int] = None) -> List[int]:
    """Derive the counter-clockwise reach vector L from the clockwise reach vector R.

    L[j] is the largest `step` such that some vertex i with (i + step) % n == j has
    R[i] >= step, i.e. the farthest rim vertex that reaches j clockwise.
    """
    n = n if n is not None else len(R)
    L = [0] * n
    for i in range(n):
        for step in range(1, R[i] + 1):
            target = (i + step) % n
            if step > L[target]:
                L[target] = step
    return L


def is_legal_neighborly_vector(R: List[int], L: List[int]) -> bool:
    """Check the neighborly-wheel legality conditions on an (R, L) pair.

    R[i], L[i] >= 1: every rim vertex reaches at least its immediate neighbors.
    R[i] + L[i] <= n - 1: combined reach can't wrap around and cover the whole rim.
    L[(i+R[i]) % n] >= R[i] and R[(i-L[i]) % n] >= L[i]: R and L must agree with
        each other at the vertices they each reach the farthest.
    R[(i+1) % n] >= R[i] - 1 and L[(i-1) % n] >= L[i] - 1: reach can drop by at
        most 1 between cyclically adjacent rim vertices ("neighborly" consistency).
    """
    n = len(R)
    if n == 0 or len(L) != n:
        return False
    for i in range(n):
        if R[i] < 1 or L[i] < 1:
            return False
        if R[i] + L[i] > n - 1:
            return False
        if L[(i + R[i]) % n] < R[i]:
            return False
        if R[(i - L[i]) % n] < L[i]:
            return False
        if R[(i + 1) % n] < R[i] - 1:
            return False
        if L[(i - 1) % n] < L[i] - 1:
            return False
    return True


def get_canonical_bracelet(R: List[int], L: List[int]) -> Tuple[int, ...]:
    """Canonical representative of (R, L) under the wheel's rotation/reflection
    symmetry group, for deduplicating rotation/reflection-equivalent vectors.

    IMPORTANT correctness note: roughly half of the original copies of this
    function (the ones consolidated here) built the "reflection" candidates from
    rotations of `reversed(R)` instead of `reversed(L)`. That is wrong in general:
    reflecting a wheel reverses its rim orientation, which turns clockwise reach
    into counter-clockwise reach, i.e. the reflected R-sequence is `reversed(L)`,
    not `reversed(R)` — L is *derived* from R via get_L_from_R and is not in
    general equal to R reversed. Using `reversed(R)` under-canonicalizes some
    truly-equivalent vectors as distinct and can, in principle, over-merge others.
    This implementation uses `reversed(L)`, which is the mathematically correct one
    (confirmed against the rotation/reflection semantics — see PROPOSED_STRUCTURE.md
    audit and the Phase 2 library-extraction notes for the two variants found).
    """
    n = len(R)
    candidates = []

    curr = list(R)
    for _ in range(n):
        candidates.append(tuple(curr))
        curr = curr[1:] + curr[:1]

    curr = list(L)[::-1]
    for _ in range(n):
        candidates.append(tuple(curr))
        curr = curr[1:] + curr[:1]

    return max(candidates)


def _backtrack_valid_R(n: int, current_R: List[int], current_L: List[int], results: List[List[int]], seen: set) -> None:
    k = len(current_R)
    if k == n:
        if is_legal_neighborly_vector(current_R, current_L):
            canonical = get_canonical_bracelet(current_R, current_L)
            if canonical not in seen:
                seen.add(canonical)
                results.append(list(current_R))
        return

    start = 1 if k == 0 else max(1, current_R[-1] - 1)
    if k == 0:
        end = n - 2
    else:
        wrap_around_limit = current_R[0] + n - k
        l_limit = n - 1 - current_L[k]
        end = min(n - 2, wrap_around_limit, l_limit)

    for val in range(start, end + 1):
        new_R = current_R + [val]
        new_L = list(current_L)
        for step in range(1, val + 1):
            target = (k + step) % n
            if step > new_L[target]:
                new_L[target] = step
        _backtrack_valid_R(n, new_R, new_L, results, seen)


def _worker_generate_valid_R(n: int, r0: int) -> List[List[int]]:
    results: List[List[int]] = []
    L0 = [0] * n
    for step in range(1, r0 + 1):
        L0[step % n] = step
    _backtrack_valid_R(n, [r0], L0, results, set())
    return results


def generate_all_valid_R(n: int, *, multiprocess: bool = False) -> List[List[int]]:
    """All legal neighborly R vectors of length n, one representative per
    rotation/reflection equivalence class.

    Uses the pruned backtracking search (bounding each new entry by the wrap-around
    and L-derived limits), which a self-test already present in the repo
    (`__check_r_generator_improvement.py`'s `compare_generators`) verified produces
    the same result set as an unpruned brute-force search, just far faster — the
    unpruned/naive variant is not carried over here. Set multiprocess=True to
    parallelize the outer loop over the first entry R[0] across CPU cores (only
    worth it for larger n; the dedup pass across workers' results still has to run
    on the canonical-bracelet form, so it's done here rather than per-worker).
    """
    if multiprocess:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            worker_results = pool.starmap(_worker_generate_valid_R, [(n, r0) for r0 in range(1, n - 1)])
        candidates = [r for batch in worker_results for r in batch]
    else:
        candidates = []
        for r0 in range(1, n - 1):
            candidates.extend(_worker_generate_valid_R(n, r0))

    seen = set()
    results = []
    for R in candidates:
        L = get_L_from_R(R)
        canonical = get_canonical_bracelet(R, L)
        if canonical not in seen:
            seen.add(canonical)
            results.append(R)
    return results
